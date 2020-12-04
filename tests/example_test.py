# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:04:35 2020
@author: suhail
"""

# from conftest import pytest, project_types

# def test_uppercase():
#     assert "loud noises".upper() == "LOUD NOISES"

# def test_reversed():
#     assert list(reversed([1, 2, 3, 4])) == [4, 3, 2, 1]

# def test_some_primes():
#     assert 37 in {
#         num
#         for num in range(1, 50)
#         if num != 1 and not any([num % div == 0 for div in range(2, num)])
#     }



# def format_data_for_display(people):
#     result = []
#     for i in people:
#         firstName = i["given_name"]
#         familyName = i["family_name"]
#         title = i["title"]
#         result.append(f"{firstName} {familyName}: {title}")
#     return result
# def test_format_data_for_display(example_people_data):    
#     assert format_data_for_display(example_people_data) == [
#         "Alfonsa Ruiz: Senior Software Engineer",
#         "Sayid Khan: Project Manager",
#     ]

# def format_data_for_excel(people):
#     result = "given,family,title\n"
#     for i in people:
#         firstName = i["given_name"]
#         familyName = i["family_name"]
#         title = i["title"]
#         result += f"{firstName},{familyName},{title}\n"
#     return result

# def test_format_data_for_excel(example_people_data):
#     assert format_data_for_excel(example_people_data) == """given,family,title
# Alfonsa,Ruiz,Senior Software Engineer
# Sayid,Khan,Project Manager
# """



# def is_palindrome(palindrome):
#     noSpace = palindrome.replace(' ','').replace('?','').lower()
#     if ((len(noSpace) == 0) or (len(noSpace) == 1)):
#         return True
#     if (noSpace[0] == noSpace[-1]):
#         return is_palindrome(noSpace[1:-1])
#     return False

# @pytest.mark.parametrize("palindrome", [
#     "",
#     "a",
#     "Bob",
#     "Never odd or even",
#     "Do geese see God?",
# ])
# def test_is_palindrome1(palindrome):
#     assert is_palindrome(palindrome)

# @pytest.mark.parametrize("non_palindrome", [
#     "abc",
#     "abab",
# ])
# def test_is_palindrome_not_palindrome(non_palindrome):
#     assert not is_palindrome(non_palindrome)
    
# @pytest.mark.parametrize("maybe_palindrome, expected_result", [
#     ("", True),
#     ("a", True),
#     ("Bob", True),
#     ("Never odd or even", True),
#     ("Do geese see God?", True),
#     ("abc", False),
#     ("abab", False),
# ])
# def test_is_palindrome(maybe_palindrome, expected_result):
#     assert is_palindrome(maybe_palindrome) == expected_result
    
    
# !pytest --cov-report term-missing --cov
# !pytest --durations=3
# !pytest --markers
# !pytest --cov-report html --cov-report xml --cov-report annotate --cov=myproj tests/
# !pytest --cov-report html --cov
# !pytest --randomly-seed=1234
# !pytest --randomly-seed=last
#    --randomly-dont-reset-seed - turn off the reset of random.seed() at the start of every test
#    --randomly-dont-reorganize - turn off the shuffling of the order of tests
# !pytest -p no:randomly


