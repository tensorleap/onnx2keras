import pytest

def pytest_addoption(parser):
    parser.addoption("--cloud_dir", action="store")
    parser.addoption("--model_name", action="store")


@pytest.fixture(scope="session")
def cloud_dir(pytestconfig):
    return pytestconfig.getoption("cloud_dir")


@pytest.fixture(scope="session")
def model_name(pytestconfig):
    return pytestconfig.getoption("model_name")
