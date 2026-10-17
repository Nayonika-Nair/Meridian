"""SSRF protection. Every URL (first request AND every redirect hop) passes through check_url().

Policy: http/https only, no credentials, allow-listed ports, and EVERY address the host resolves to must be a
globally routable unicast address. Loopback, RFC1918, link-local (incl. 169.254.169.254 cloud metadata), CGNAT,
multicast, reserved, unspecified and IPv4-mapped/6to4/NAT64 forms of those are refused.

Known limit: the check resolves the name, then the HTTP client resolves it again (DNS-rebinding window).
For stronger isolation run the collector behind an egress proxy that enforces the same rule.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

ALLOWED_SCHEMES = ("http", "https")
DEFAULT_PORTS = (80, 443)
_NUMERICISH = re.compile(r"^[0-9a-fA-Fx.]+$")
_BAD_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".corp", ".intranet", ".arpa")


class UrlBlocked(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


@dataclass(frozen=True)
class Target:
    url: str
    scheme: str
    host: str
    port: int
    ips: tuple


def default_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({i[4][0] for i in infos})


def ip_is_public(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    if isinstance(a, ipaddress.IPv6Address):
        if a.ipv4_mapped is not None:
            return ip_is_public(str(a.ipv4_mapped))
        if a.sixtofour is not None:
            return ip_is_public(str(a.sixtofour))
        if a in ipaddress.ip_network("64:ff9b::/96"):                    # NAT64 embeds a v4 address in the last 32 bits
            return ip_is_public(str(ipaddress.IPv4Address(int(a) & 0xFFFFFFFF)))
    return bool(a.is_global and not a.is_multicast)


def check_url(url: str, *, resolver=None, allow_private: bool = False, ports=DEFAULT_PORTS, max_len: int = 2048) -> Target:
    if not isinstance(url, str) or not url or len(url) > max_len:
        raise UrlBlocked("bad_url", "empty or too long")
    if any(ord(c) < 33 or ord(c) == 127 for c in url):
        raise UrlBlocked("bad_url", "control characters or whitespace")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise UrlBlocked("bad_url", str(exc)) from exc
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UrlBlocked("scheme_not_allowed", parts.scheme or "none")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise UrlBlocked("credentials_in_url")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise UrlBlocked("bad_url", "no host")
    port = port or (443 if parts.scheme.lower() == "https" else 80)
    if not allow_private and port not in ports:
        raise UrlBlocked("port_not_allowed", str(port))
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is None:
        if _NUMERICISH.match(host):                                       # 2130706433, 0x7f.1, 017700000001 ...
            raise UrlBlocked("obfuscated_ip", host)
        if not allow_private and (host == "localhost" or "." not in host or host.endswith(_BAD_SUFFIXES)):
            raise UrlBlocked("internal_hostname", host)
        try:
            host.encode("idna")
        except UnicodeError as exc:
            raise UrlBlocked("bad_url", "invalid hostname") from exc
        try:
            ips = list((resolver or default_resolver)(host, port))
        except (OSError, UnicodeError) as exc:
            raise UrlBlocked("dns_failure", host) from exc
    else:
        ips = [str(literal)]
    if not ips:
        raise UrlBlocked("dns_failure", host)
    if not allow_private:
        for ip in ips:
            if not ip_is_public(ip):
                raise UrlBlocked("non_public_address", f"{host} -> {ip}")
    return Target(url=url, scheme=parts.scheme.lower(), host=host, port=port, ips=tuple(ips))
