from dataclasses import dataclass 

@dataclass
class TestClass:
    constants = 0 
    list = []

test = TestClass()

test2 = TestClass()
test.constants += 8 

print(test.constants)
print(test2.constants)

