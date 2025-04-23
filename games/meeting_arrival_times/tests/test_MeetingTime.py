import numpy as np
from games.meeting_arrival_times.MeetingTime import MeetingTime


class TestMeetingTime:
    def test_diffusion(self):
        nu = 1
        scheduled_time = 12
        meeting_time = MeetingTime(scheduled_time, 0.8, nu)
        assert meeting_time.diffusion(None, None, None) == np.sqrt(2 * nu)

    def test_loss(self):
        nu = 1
        scheduled_time = 12
        meeting_time = MeetingTime(scheduled_time, 0.8, nu)
        assert meeting_time.personal_inconvenience(11, 12) == 0
        assert meeting_time.personal_inconvenience(12, 11) == 1
        assert meeting_time.reputation_effect(scheduled_time-1) == 0
        assert meeting_time.reputation_effect(scheduled_time+1) == 1
        assert meeting_time.waiting_time(11, 10) == 0
        assert meeting_time.waiting_time(11, 12) == 1


