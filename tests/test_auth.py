import base64

from chicken_tracker import auth as A


def hdr(user, pw):
    return "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()


def test_login_and_lockout(tmp_path, monkeypatch):
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    f = tmp_path / "auth.json"
    g = A.Gate(f)
    assert not g.ready and not g.check(hdr("hen", "x"), "1.2.3.4")  # no password set: nobody gets in
    A.set_password("hen", "correct horse battery", f)
    assert g.ready and (f.stat().st_mode & 0o777) == 0o600
    assert "correct horse" not in f.read_text()  # only a hash is stored
    assert g.check(hdr("hen", "correct horse battery"), "1.2.3.4")
    assert not g.check(hdr("hen", "wrong"), "5.6.7.8") and not g.check(hdr("fox", "correct horse battery"), "5.6.7.8")
    assert not g.check(None, "9.9.9.9") and not g.locked("9.9.9.9")  # the first, header-less request isn't a fail
    for _ in range(10):
        g.check(hdr("hen", "guess"), "5.6.7.8")
    assert g.locked("5.6.7.8") and not g.locked("1.2.3.4")
