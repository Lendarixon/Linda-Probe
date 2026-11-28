"""Corporate LDAP/AD authentication for LocalDocs.

Maps directory groups to LocalDocs groups. Real bind via ldap3;
unit-tested with duck-typed fake connections (no domain needed for tests).
"""
import os

DEFAULT_GROUP_ATTR = "memberOf"


def short_name(dn, group_map=None):
    """'CN=Auditors,OU=Groups,DC=corp' -> mapped name or 'Auditors'."""
    if group_map:
        for k, v in group_map.items():
            if k.lower() == dn.lower():
                return v
    for part in dn.split(","):
        if part.strip().upper().startswith("CN="):
            return part.strip()[3:]
    return dn


def groups_from_entry(entry, group_attr=DEFAULT_GROUP_ATTR, group_map=None):
    raw = []
    try:
        vals = entry[ group_attr ].values if hasattr(entry, "__getitem__") else []
        raw = list(vals or [])
    except (KeyError, AttributeError, TypeError):
        raw = []
    if not raw and hasattr(entry, "memberOf"):
        try:
            raw = list(entry.memberOf.values or [])
        except (AttributeError, TypeError):
            raw = []
    return {short_name(d, group_map) for d in raw if d}


class LDAPAuthenticator:
    def __init__(self, host, port=636, use_ssl=True, bind_dn="", bind_pw="",
                 base_dn="", user_filter="(sAMAccountName={login})",
                 group_attr=DEFAULT_GROUP_ATTR, group_map=None, timeout=10):
        self.host = host
        self.port = port
        self.use_ssl = use_ssl
        self.bind_dn = bind_dn
        self.bind_pw = bind_pw
        self.base_dn = base_dn
        self.user_filter = user_filter
        self.group_attr = group_attr
        self.group_map = group_map or {}
        self.timeout = timeout

    @classmethod
    def from_config(cls, cfg):
        return cls(host=cfg.get("host", ""), port=int(cfg.get("port", 636)),
                   use_ssl=bool(cfg.get("use_ssl", True)),
                   bind_dn=cfg.get("bind_dn", ""), bind_pw=cfg.get("bind_pw", ""),
                   base_dn=cfg.get("base_dn", ""),
                   user_filter=cfg.get("user_filter", "(sAMAccountName={login})"),
                   group_attr=cfg.get("group_attr", DEFAULT_GROUP_ATTR),
                   group_map=cfg.get("group_map", {}),
                   timeout=int(cfg.get("timeout", 10)))

    def _connect(self):
        import ldap3
        srv = ldap3.Server(self.host, port=self.port, use_ssl=self.use_ssl,
                           connect_timeout=self.timeout)
        return ldap3.Connection(srv, user=self.bind_dn, password=self.bind_pw,
                                auto_bind=True, receive_timeout=self.timeout)

    def authenticate(self, login, password):
        """Bind as service -> find user -> verify password -> groups. None on failure."""
        import ldap3
        login = (login or "").strip().replace("*", "").replace("(", "").replace(")", "")
        if not login or not password:
            return None
        try:
            conn = self._connect()
        except Exception:
            return None
        try:
            filt = self.user_filter.format(login=login)
            if not conn.search(self.base_dn, filt, attributes=[self.group_attr, "distinguishedName"]):
                return None
            if not conn.entries:
                return None
            user = conn.entries[0]
            user_dn = str(getattr(user, "entry_dn", "") or user.entry_dn)
            try:
                probe = ldap3.Connection(conn.server, user=user_dn, password=password,
                                         auto_bind=True, receive_timeout=self.timeout)
                probe.unbind()
            except Exception:
                return None
            return groups_from_entry(user, self.group_attr, self.group_map)
        except Exception:
            return None
        finally:
            try:
                conn.unbind()
            except Exception:
                pass


def ldap_enabled():
    return bool(os.environ.get("LOCALDOCS_LDAP") == "1")
