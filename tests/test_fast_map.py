import pytest
from malva.fast_map import FastMapOfMap

class TestFastMap:
    @pytest.fixture
    def fmap(self):
        """Create a fresh FastMapOfMap instance for each test."""
        return FastMapOfMap()

    def test_add_and_get(self, fmap):
        key = 12345
        value = 67890
        fmap.add(key, value)
        assert key in fmap
        assert value in fmap[key]

    def test_multiple_values(self, fmap):
        key = 12345
        values = [100, 200, 300]
        for value in values:
            fmap.add(key, value)
        
        retrieved = fmap[key]
        assert len(retrieved) == len(values)
        assert all(v in retrieved for v in values)

    def test_clear(self, fmap):
        fmap.add(1, 100)
        fmap.add(2, 200)
        assert len(fmap) == 2
        fmap.clear()
        assert len(fmap) == 0

    def test_nonexistent_key(self, fmap):
        with pytest.raises(KeyError):
            _ = fmap[999]

    def test_extend(self, fmap):
        key = 12345
        values = [100, 200, 300]
        fmap.extend(key, values)
        retrieved = fmap[key]
        assert len(retrieved) == len(values)
        assert all(v in retrieved for v in values)