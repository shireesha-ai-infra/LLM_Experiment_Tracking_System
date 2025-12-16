def simple_quality_score(response):
    length_score = min(len(response.split()) / 50, 1.0)
    return round(length_score * 5, 2)