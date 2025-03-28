RESTYPES = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
RESTYPE_ORDER = {restype: i for i, restype in enumerate(RESTYPES)}
RESTYPES_WITH_X = RESTYPES + ['X']
RESTYPE_ORDER_WITH_X = {restype: i for i, restype in enumerate(RESTYPES_WITH_X)}
