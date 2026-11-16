SAMPLE_NO_CONFLICT = """def foo():
    return 1
"""

SAMPLE_SINGLE_CONFLICT = """\
line1
<<<<<<< HEAD
ours line
=======
theirs line
>>>>>>> target-branch
line5
"""

SAMPLE_TWO_CONFLICTS = """\
<<<<<<< HEAD
ours1
=======
theirs1
>>>>>>> target
middle
<<<<<<< HEAD
ours2
=======
theirs2
>>>>>>> target
"""

SAMPLE_EMPTY_SIDES = """\
<<<<<<< HEAD
=======
theirs only
>>>>>>> target
"""

SAMPLE_MALFORMED = """\
<<<<<<< HEAD
no closer
"""
