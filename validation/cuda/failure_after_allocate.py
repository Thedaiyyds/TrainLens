"""Save direct readings before an intentional Python training failure."""

if __package__:
    from .allocate_peak import write_reference
else:
    from allocate_peak import write_reference

if __name__ == "__main__":
    write_reference()
    raise RuntimeError("intentional validation failure")
