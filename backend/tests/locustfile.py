from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 2)

    @task(3)
    def check_health(self):
        self.client.get("/health")

    @task(1)
    def simulate_heavy_request(self):
        self.client.get("/health")
