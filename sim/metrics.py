def calculate_metrics(shortage, wastage):

    return {
        "shortage": shortage,
        "wastage": wastage,
        "efficiency": max(0, 100 - shortage - wastage)
    }