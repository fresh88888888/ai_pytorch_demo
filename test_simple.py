import pytest

def inc(x):
    return x + 1

def test_answer():
    assert inc(4) == 5

def f():
    raise SystemExit(1)

def test_mytest():
    with pytest.raises(SystemExit):
        f()

def f1():
    raise ExceptionGroup(
        'Group Message',
        [RuntimeError(),],
    )

def test_exception_in_group():
    with pytest.raises(ExceptionGroup) as excinfo:
        f1()
    assert excinfo.group_contains(RuntimeError)
    assert not excinfo.group_contains(TypeError)


def test_needsfiles(tmp_path):
    print(tmp_path)
    assert 0

class TestClass:
    def test_one(self):
        x = 'this'
        assert 'h'in x
        
    def test_two(self):
        x = 'hello'
        assert hasattr(x, 'he')

class TestClassDemoInstance():
    value = 0
    def test_one(self):
        self.value = 1
        assert self.value == 1
    
    def test_two(self):
        assert self.value == 1

class TestTempCase():
    def test_needsfiles(self, tmp_path):
        print(tmp_path)
        assert 0