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


# Config keys on `cfg.student` that are training recipe, not model constructor kwargs.
# Lives here (not in the trainer) so deploy code can rebuild a student without importing
# the training stack — which pulls in PennyLane, and an export path must not need that.
STUDENT_RECIPE_KEYS = {
    "name", "epochs", "patience", "optimizer", "lr", "weight_decay", "force_batch_stats_only",
}


def build_student_from_cfg(student_cfg, num_classes):
    kwargs = {k: v for k, v in student_cfg.items() if k not in STUDENT_RECIPE_KEYS}
    return build_student(student_cfg.name, num_classes=num_classes, **kwargs)
