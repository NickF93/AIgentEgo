import aigentego


def test_package_imports() -> None:
    assert isinstance(aigentego.__version__, str)
