# an org turns optional modules on or off. core modules are always on and cannot be switched.
# switch state lives in org.config["modules"]; a missing key means ON (existing orgs keep features).
# only an explicit False turns a module off. a disabled module answers 404, not 403.
# a blueprint can gate every route, and single endpoints can override with their own module.
# from platform: modules/organizations/service.py module_enabled(), new_org_switches(), OPTIONAL_MODULES.
# from platform: modules/registry.py Mount(module, endpoint_modules) and _gate() before_request hook.
# explain: why 404 and not 403 for an off module? what does it hide?
# explain: why is "missing key means on" a migration decision and not just a default?

from dataclasses import dataclass, field

OPTIONAL = {"points", "storefront", "godfather"}


def module_enabled(config: dict | None, name: str) -> bool:
    # names not in OPTIONAL are always on. otherwise config.get("modules", {}).get(name, True) is not False.
    # config may be None or have modules=None; do not crash.
    pass


def new_org_switches(on_by_default: set[str]) -> dict[str, bool]:
    # a new org: True for names in on_by_default, False for every other optional module.
    pass


@dataclass(frozen=True)
class Mount:
    module: str | None = None
    endpoint_modules: dict[str, str] = field(default_factory=dict)


def gate(mount: Mount, config: dict | None, endpoint: str) -> tuple[int, str] | None:
    # module = endpoint_modules.get(endpoint, mount.module). no module -> None (allowed).
    # module off -> (404, message naming the module). on -> None.
    pass


if __name__ == '__main__':
    # assert: empty config keeps everything on, False switches one module off only,
    # an endpoint override gates a leaderboard route by "points" inside an ungated blueprint,
    # and new_org_switches returns exactly one key per optional module.
    pass
