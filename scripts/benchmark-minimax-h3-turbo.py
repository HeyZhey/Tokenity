#!/usr/bin/env python3
"""Opt-in Swift Store -> Tokenity gateway -> native H3 -> AVFoundation matrix.
Uses the existing native benchmark's timing/resource parsers; no new dependencies.
"""
from __future__ import annotations
import argparse, array, hashlib, importlib.util, json, math, os, pathlib, shutil
import subprocess, sys, threading, time, urllib.request, wave

ROOT = pathlib.Path(__file__).resolve().parents[1]


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(16 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-repo', type=pathlib.Path, required=True)
    p.add_argument('--model', type=pathlib.Path, required=True)
    p.add_argument('--binary', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--configs', default='turbo6,baseline,turbo4,turbo8')
    p.add_argument('--warm-runs', type=int, default=3)
    p.add_argument('--agent-port', type=int, default=19100)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('native_h3_benchmark', args.native_repo / 'scripts/benchmark-h3-turbo.py')
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    env = os.environ.copy()
    env.update({'PYTHONPATH': str(ROOT), 'TOKENITY_CODE_ROOT': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1',
                'MINIMAX_H3_BENCH': '1', 'MINIMAX_H3_VALIDATE_OUTPUT': '1',
                'MINIMAX_H3_STEP_CACHE': '0', 'MINIMAX_H3_ATTN_BCAST': '0'})
    host = {'chip': native.command(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string']),
            'memory_bytes': native.command(['/usr/sbin/sysctl', '-n', 'hw.memsize']),
            'os': native.command(['/usr/bin/sw_vers']), 'binary': str(args.binary.resolve()),
            'binary_sha256': sha256(args.binary), 'model': str(args.model.resolve()),
            'temperature_c': None, 'gpu_clock_mhz': None,
            'environment': {key: env[key] for key in ('MINIMAX_H3_BENCH', 'MINIMAX_H3_VALIDATE_OUTPUT', 'MINIMAX_H3_STEP_CACHE', 'MINIMAX_H3_ATTN_BCAST')},
            'cold_definition': 'First generation in each process started by TokenityStore through Node Agent; OS/Metal disk caches are retained.',
            'warm_definition': 'Subsequent generations in the same process; existing staged per-request model loading is retained.',
            'timing_definition': 'inference_e2e_s is gateway response through Swift complete-event parsing; ui_save_s is AVFoundation artifact saving; ui_total_s includes both. Direct backend HTTP time is not measured.',
            'before': native.snapshot(os.getpid())}
    host['model_artifacts'] = {path.name: {'size_bytes': path.stat().st_size, 'sha256': sha256(path)}
                               for path in sorted(args.model.iterdir()) if path.name in {
                                   'config.json', 'transformer.safetensors', 'text_encoder.safetensors',
                                   'video_vae.safetensors', 'audio_vae.safetensors', 'turbo_lora.safetensors'}}
    write(args.out/'host.json', host)
    logs = args.out/'native-logs'; logs.mkdir()
    # Same installed Node Agent application, isolated port and task-owned logs.
    program = ('import logging, pathlib, uvicorn; '
               'from tokenity.node_agent.agent import create_app; '
               'from tokenity.process.supervisor import RoleSupervisor; '
               'logging.basicConfig(level=logging.INFO); '
               f'uvicorn.run(create_app(supervisor=RoleSupervisor(log_dir=pathlib.Path({str(logs)!r}))), host="127.0.0.1", port={args.agent_port})')
    stop = threading.Event(); samples = {}; last = None
    def monitor():
        nonlocal last
        while not stop.wait(1):
            try:
                active = json.loads((args.out/'active-run.json').read_text())
                if 'pid' not in active: continue
                directory = pathlib.Path(active['directory'])
                if directory != last and last is not None:
                    write(last/'resources.json', samples[str(last)])
                last = directory
                samples.setdefault(str(directory), []).append(native.snapshot(active['pid']))
                write(directory/'resources.json', samples[str(directory)])
            except (OSError, ValueError): pass
    with (args.out/'agent.log').open('w') as agentlog:
        agent = subprocess.Popen([sys.executable, '-u', '-c', program], cwd=ROOT, env=env, stdout=agentlog, stderr=subprocess.STDOUT)
        try:
            for _ in range(60):
                if agent.poll() is not None: raise RuntimeError((args.out/'agent.log').read_text())
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{args.agent_port}/v1/node/info', timeout=3).close(); break
                except Exception: time.sleep(1)
            else: raise TimeoutError('Node Agent did not become ready')
            env.update({'TOKENITY_H3_LIVE_OUT': str(args.out), 'TOKENITY_H3_LIVE_BINARY': str(args.binary),
                        'TOKENITY_H3_LIVE_MODEL': str(args.model), 'TOKENITY_H3_LIVE_CONFIGS': args.configs,
                        'TOKENITY_H3_LIVE_WARM_RUNS': str(args.warm_runs),
                        'TOKENITY_H3_LIVE_AGENT': f'http://127.0.0.1:{args.agent_port}'})
            thread = threading.Thread(target=monitor, daemon=True); thread.start()
            with (args.out/'swift-live.log').open('w') as log:
                result = subprocess.run(['swift', 'test', '--skip-build', '--package-path', str(ROOT/'apps/TokenityControl'),
                                         '--filter', 'VideoGenerationLiveTests/testLiveGatewayMatrix'],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'Swift live test failed; inspect {args.out / "swift-live.log"}')
        finally:
            stop.set()
            if 'thread' in locals(): thread.join()
            agent.terminate()
            try: agent.wait(30)
            except subprocess.TimeoutExpired: agent.kill(); agent.wait()
    results = []
    for path in sorted(args.out.glob('*/*/ui-result.json')):
        folder = path.parent; ui = json.loads(path.read_text())
        artifact = pathlib.Path(ui['artifact_directory']); metadata = json.loads((artifact/'metadata.json').read_text())
        metrics = native.metrics((folder/'server.log').read_text())
        resources = json.loads((folder/'resources.json').read_text())
        request = json.loads((folder/'request.json').read_text())
        assert len(metrics['step_times_s']) == request['steps'], 'Native steps must match UI request'
        assert metrics['all_finite'] is True
        if request['turbo']:
            assert 'turbo=true' in metrics['speed_config']
            assert '259/259' in (folder/'server.log').read_text()
        assert 'step-cache 0.000, attn-broadcast k=0' in metrics['speed_config']
        assert metadata['turbo'] == request['turbo'] and metadata['steps'] == request['steps'] and metadata['fast'] is False
        movie = artifact/'generation.mov'
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(movie), '-f', 'null', '-'], check=True)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(movie)]))
        write(folder/'ffprobe.json', probe)
        with wave.open(str(artifact/'audio.wav'), 'rb') as audio:
            audio_duration = audio.getnframes()/audio.getframerate()
            pcm = array.array('h', audio.readframes(audio.getnframes()))
        video_duration = metadata['frames']/metadata['fps']
        assert abs(audio_duration-video_duration) < 1/metadata['fps']
        metrics.update(ui)
        metrics.update({'request': request, 'inference_e2e_s': ui['gateway_stream_s'],
                        'e2e_including_encoding_s': ui['ui_total_s'], 'resources': resources,
                        'rss_peak_bytes': max(x['rss_bytes'] or 0 for x in resources),
                        'rss_before_bytes': resources[0]['rss_bytes'], 'rss_after_bytes': resources[-1]['rss_bytes'],
                        'swap_growth_bytes': resources[-1]['swap_used_bytes']-resources[0]['swap_used_bytes'],
                        'thermal_statuses': list(dict.fromkeys(x['thermal'] for x in resources if x['thermal'])),
                        'media': {'movie': str(movie), 'wav': str(artifact/'audio.wav'), 'decode_ok': True,
                                  'video_duration_s': video_duration, 'audio_duration_s': audio_duration,
                                  'av_duration_delta_s': audio_duration-video_duration,
                                  'rgb_sha256': sha256(artifact/'video.rgb'), 'pcm_sha256': sha256(artifact/'audio.pcm'),
                                  'audio_rms': math.sqrt(sum(x*x for x in pcm)/len(pcm))/32768,
                                  'audio_clip_fraction': sum(abs(x)>=32767 for x in pcm)/len(pcm)}})
        write(folder/'result.json', metrics); results.append(metrics)
    host['after'] = native.snapshot(os.getpid())
    write(args.out/'results.json', {'host': host, 'summary': native.summarize(results), 'runs': results})
    print(args.out/'results.json', flush=True)


if __name__ == '__main__': main()
