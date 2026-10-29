def calculate_points(results):
    count = 0
    win = "w"
    draw = "d"
    for result in results:
        if result.lower() == win:
            count += 3
        elif result.lower() == draw:
            count += 1
    return count


def last_five(results):
    return calculate_points(results[-5:])
