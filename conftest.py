"""Function to log pytest outputs."""

import pytest

import time


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Log the pytest outputs.

    Args:
        item: item
        call: call
    """
    outcome = yield  # Get the outcome of the test
    result = outcome.get_result()  # Get the report object
    timestamp = time.strftime("%Y_%m_%d-%H:%M:%S")
    log_file = f"tests/.testlogs/test_{timestamp}_{result.outcome.upper()}.log"  # The name of the log file
    if result.when == "call":  # Only log the results of the test function call
        try:
            with open(log_file, "a") as f:  # Open the log file in append mode
                f.write(
                    f"[{result.outcome}] "
                    + result.nodeid
                    + " "
                    + result.outcome
                    + " "
                    + str(result.duration)
                    + "\n"
                )  # Write the node id, outcome, and duration of the test
                if result.longrepr is not None:
                    f.write("Error :\n")
                    f.write(str(result.longrepr) + "\n\n")
        except Exception as e:
            print("Error", e)  # Handle any errors in writing to the log file
