from collections import Counter


text1 = "anagram"
text2 = "na gaa rm"

def is_anagram_counter(s1, s2):
    # Remove spaces and convert to lowercase
    s1 = s1.replace(" ", "").lower()
    s2 = s2.replace(" ", "").lower()
    return Counter(s1) == Counter(s2)

def is_anagram_manual(s1, s2):
    if len(s1) != len(s2):
        return False
    freq = {}
    for c in s1:
        freq[c] = freq.get(c, 0) + 1   # increment for s1
    for c in s2:
        freq[c] = freq.get(c, 0) - 1   # decrement for s2
        if freq[c] < 0:
            return False                # s2 has a char s1 doesn't
    return True

# print(is_anagram_counter(text1, text2))
# print(is_anagram_manual(text1, text2))

assert is_anagram_counter("anagram", "nagaram") == True
assert is_anagram_counter("rat", "car") == False
assert is_anagram_counter("a", "a") == True
assert is_anagram_counter("ab", "a") == False

assert is_anagram_manual("anagram", "nagaram") == True
assert is_anagram_manual("rat", "car") == False

print("All tests passed.")