from apscheduler.schedulers.background import BackgroundScheduler
import logging

log = logging.getLogger(__name__)


class DataScheduler:
    def __init__(self):
        self._sched = BackgroundScheduler()
        self.jobs = {}

    def add_job(self, func, interval_secs, job_id):
        self.jobs[job_id] = self._sched.add_job(func, 'interval', seconds=interval_secs, id=job_id)
        log.debug(f"job added: {job_id}")

    def add_cron_job(self, func, hour, minute, job_id):
        self.jobs[job_id] = self._sched.add_job(func, 'cron', hour=hour, minute=minute, id=job_id)

    def start(self):
        if not self._sched.running:
            self._sched.start()
            log.info("scheduler started")

    def stop(self):
        if self._sched.running:
            self._sched.shutdown()
            log.info("scheduler stopped")

    def list_jobs(self):
        return self._sched.get_jobs()

    def remove_job(self, job_id):
        if job_id in self.jobs:
            self._sched.remove_job(job_id)
            del self.jobs[job_id]
