
def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations + 1):
        j = i * param1 - param2
        result -= 1 / j
        j = i * param1 + param2
        result += 1 / j
    return result

result = calculate(200000000, 4, 1) * 4
print(f"Result: {result:.12f}")
