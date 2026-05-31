def hasDuplicate(nums):
    # your code here
    num_set = set()
    for n in nums:
        if n in num_set:
            return True
        num_set.add(n)
    return False


if __name__ == "__main__":
    print(hasDuplicate([1, 2, 3, 1]))  # True
    print(hasDuplicate([1, 2, 3, 4]))  # False