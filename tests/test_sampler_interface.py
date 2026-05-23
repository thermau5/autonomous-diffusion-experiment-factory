import pytest

from autonomous_diffusion.samplers import get_sampler, list_samplers


def test_edm_samplers_registered():
    assert "edm_euler" in list_samplers()
    assert "edm_heun" in list_samplers()


def test_get_unknown_sampler_raises():
    with pytest.raises(KeyError):
        get_sampler("not_a_real_sampler")


def test_sampler_instance_has_id():
    for sid in ["edm_euler", "edm_heun"]:
        s = get_sampler(sid)
        assert s.id == sid
