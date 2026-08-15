STUDENTS = {}
TEACHERS = {}


def register_student(name):
    def _wrap(cls):
        STUDENTS[name] = cls
        return cls

    return _wrap


def register_teacher(name):
    def _wrap(cls):
        TEACHERS[name] = cls
        return cls

    return _wrap


def build_teacher(name, **kwargs):
    return TEACHERS[name](**kwargs)


def build_student(name, **kwargs):
    return STUDENTS[name](**kwargs)
