"""Actor-side client: standard library only, no evaluator or reference dependency."""
import json
import urllib.request
import uuid


class Client:
    def __init__(self, endpoint='http://127.0.0.1:8810', timeout=60):
        self.endpoint, self.timeout = endpoint.rstrip('/'), timeout

    def request(self, path, payload):
        req = urllib.request.Request(self.endpoint+path, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.load(response)

    def create(self, task_id, evidence='image', mode='evaluation'):
        return self.request('/episodes', dict(task_id=task_id, evidence=evidence, mode=mode))

    def resume(self, episode_id):
        return self.request('/resume', dict(episode_id=episode_id))

    def step(self, episode_id, expected_step, action, request_id):
        # Persist request_id BEFORE sending; reuse it after transport interruption.
        return self.request('/step', dict(episode_id=episode_id, expected_step=expected_step,
                                         action=action, request_id=request_id))

    @staticmethod
    def new_request_id():
        return uuid.uuid4().hex
