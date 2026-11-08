import hashlib, json, os, re, subprocess, tempfile, time
from pathlib import Path

DISABLED_FEATURES = ('shell_tool', 'unified_exec', 'view_image', 'apps', 'plugins', 'browser_use', 'browser_use_external', 'computer_use', 'in_app_browser', 'multi_agent', 'multi_agent_v2', 'memories', 'hooks', 'image_generation')

def command(binary, directory, model, effort, images):
    """Same restricted CLI flags as legacy baseline, without importing its env."""
    args = [binary, 'exec', '--ignore-user-config', '--ephemeral', '--skip-git-repo-check', '--sandbox', 'read-only', '--model', model, '--json', '--cd', str(directory), '-c', f'model_reasoning_effort={json.dumps(effort)}', '-c', 'web_search="disabled"', '-c', 'project_doc_max_bytes=0']
    # Same OpenAI Codex authentication/endpoint; start on HTTPS directly.
    # This alias changes transport availability, never model/provider fallback.
    args += ['-c', 'model_provider="batch4_https"',
             '-c', 'model_providers.batch4_https.name="OpenAI HTTPS"',
             '-c', 'model_providers.batch4_https.base_url="https://chatgpt.com/backend-api/codex"',
             '-c', 'model_providers.batch4_https.wire_api="responses"',
             '-c', 'model_providers.batch4_https.requires_openai_auth=true',
             '-c', 'model_providers.batch4_https.supports_websockets=false']
    for feature in DISABLED_FEATURES:
        args += ['--disable', feature]
    args += ['--enable', 'skip_host_skill_discovery', '--output-schema', str(directory / 'schema.json'), '-o', str(directory / 'response.json')]
    for image in images:
        args += ['--image', str(image)]
    return args + ['-']

def audit_events(stdout):
    """Reject tools/errors/incomplete turns; return usage, never model reasoning."""
    usage, complete = ({}, False)
    for line in stdout.splitlines():
        event = json.loads(line)
        if event['type'] in ('turn.failed', 'error'):
            raise RuntimeError('Codex turn failed')
        if event['type'] == 'turn.completed':
            usage, complete = (event.get('usage', {}), True)
        item = event.get('item', {})
        if item and item.get('type') not in ('agent_message', 'reasoning', 'error'):
            raise RuntimeError('Non-message activity invalidates the restricted baseline')
    if not complete:
        raise RuntimeError('Codex turn did not complete')
    return usage

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _event_summary(stdout: str) -> dict:
    counts, tools = ({}, 0)
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
            name = event.get('type', 'other')
            name = name if isinstance(name, str) and re.fullmatch('[a-zA-Z0-9_.-]{1,64}', name) else 'other'
            counts[name] = counts.get(name, 0) + 1
            item = event.get('item', {})
            if isinstance(item, dict) and item and (item.get('type') not in ('agent_message', 'reasoning', 'error')):
                tools += 1
        except (ValueError, TypeError, AttributeError):
            counts['unparseable'] = counts.get('unparseable', 0) + 1
    return {'valid': False, 'tool_calls': tools, 'event_counts': counts, 'failure': ''}

class DiscoveryCodexPolicy:
    """Fresh restricted CLI context; its actual input inventory is recorded."""

    def __init__(self, model='gpt-6.1-sol', effort='low', timeout=120.0, binary='codex'):
        self.model, self.effort, self.timeout, self.binary = (model, effort, timeout, binary)

    def call(self, prompt: str, images: list[Path], schema: dict) -> dict:
        started = time.perf_counter()
        response = {'status': 'invocation_failed', 'failure': 'cli_invocation', 'usage': {}, 'audit': {'valid': False, 'tool_calls': 0, 'event_counts': {}, 'failure': 'cli_invocation'}, 'completion_text': '', 'cli_arguments': [], 'cli_input_files': [], 'cli_image_count': 0, 'cli_input_sha256': {}}
        with tempfile.TemporaryDirectory(prefix='visual-discovery-codex-', dir='/tmp') as temporary:
            directory = Path(temporary)
            (directory / 'schema.json').write_bytes(json.dumps(schema).encode())
            image_paths = []
            for index, source in enumerate(images):
                destination = directory / f'image_{index:02d}.png'
                destination.write_bytes(source.read_bytes())
                image_paths.append(destination)
            arguments = command(self.binary, directory, self.model, self.effort, image_paths)
            response.update(cli_arguments=arguments, cli_input_files=sorted((path.name for path in directory.iterdir())), cli_image_count=arguments.count('--image'), cli_input_sha256={path.name: sha(path.read_bytes()) for path in directory.iterdir()})
            expected_files = sorted(['schema.json'] + [path.name for path in image_paths])
            if response['cli_input_files'] != expected_files or response['cli_image_count'] != len(images):
                response['failure'] = 'input_inventory'
                return response
            try:
                process = subprocess.run(arguments, input=prompt, capture_output=True, text=True, timeout=self.timeout, check=False, env={key: value for key, value in os.environ.items() if key != 'OPENROUTER_API_KEY'})
                response['events'] = safe_events(process.stdout)
                response['audit'] = _event_summary(process.stdout)
                response['stderr_summary'] = scrub(process.stderr[-3000:])
                if process.returncode:
                    response.update(status='invocation_failed', failure='cli_exit')
                else:
                    try:
                        response['usage'] = audit_events(process.stdout)
                        response['audit']['valid'] = True
                        response.update(status='ok', failure='')
                    except (RuntimeError, ValueError, KeyError, TypeError):
                        reason = 'tool_activity' if response['audit']['tool_calls'] else 'event_audit'
                        response.update(status='audit_failed', failure=reason)
                        response['audit']['failure'] = reason
            except subprocess.TimeoutExpired as error:
                prefix = error.stdout or b''
                response['events'] = safe_events(prefix.decode(errors='replace') if isinstance(prefix, bytes) else prefix)
                response.update(status='timeout', failure='timeout')
            except (OSError, ValueError):
                pass
            completion = directory / 'response.json'
            if completion.exists():
                try:
                    response['completion_text'] = completion.read_bytes().decode()
                except (OSError, UnicodeError):
                    response.update(status='missing_completion', failure='missing_completion')
            elif response['status'] == 'ok':
                response.update(status='missing_completion', failure='missing_completion')
        response['wall_seconds'] = time.perf_counter() - started
        return response

def scrub(text):
    text = re.sub(r'sk-[A-Za-z0-9_-]+', '[redacted]', text)
    text = re.sub(r'(?i)Bearer\s+\S+', 'Bearer [redacted]', text)
    return re.sub(r'https?://\S+', '[URL omitted]', text)

def safe_events(stdout):
    result = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
            item = event.get('item', {})
            if item.get('type') == 'reasoning':
                continue
            result.append(json.loads(scrub(json.dumps(event))))
        except (ValueError, TypeError, AttributeError):
            result.append({'type': 'unparseable'})
    return result
