from games.meeting_arrival_times import arrival_times_example, arrival_times_nODE


def test_nODE():
    arrival_times_nODE.main(1, True)


def test_example():
    arrival_times_example.main(True)
