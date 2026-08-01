"""Pipeline stages.

Each stage is a module exposing a single public ``run(...)`` function and an
``IS_STUB`` flag. Every ``run`` takes a :class:`~videodistill.profile.DomainProfile`
so domain behaviour stays out of the code. Stages never import one another; they
communicate only through the typed artifacts in :mod:`videodistill.models` read
from / written to the job directory.
"""
