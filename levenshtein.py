def sequence_derive(cost_mat, num_rows, num_col):
    i = num_rows - 1
    j = num_col - 1
    sequence = []
    while i >= 1:
        # pdb.set_trace()
        if j == 0: break
        if ((cost_mat[i - 1][j - 1] <= cost_mat[i - 1][j]) and (cost_mat[i - 1][j - 1] <= cost_mat[i][j - 1]) and (
            cost_mat[i][j] == cost_mat[i - 1][j - 1])):
            sequence.append("C")
            i = i - 1
            j = j - 1
        elif (cost_mat[i][j] != cost_mat[i - 1][j - 1]) and (cost_mat[i - 1][j - 1] <= cost_mat[i - 1][j]) and (
            cost_mat[i - 1][j - 1] <= cost_mat[i][j - 1]) and (j - 1 >= 0):
            sequence.append("S")
            i = i - 1
            j = j - 1
        elif (cost_mat[i - 1][j] < cost_mat[i][j - 1]) and (cost_mat[i - 1][j] < cost_mat[i - 1][j - 1]):
            sequence.append("I")
            i = i - 1
        else:
            sequence.append("D")
            j = j - 1

    if i == 0 and j != 0:
        while j >= 1:
            sequence.append("D")
            j = j - 1

    elif j == 0 and i != 0:
        while i >= 0:
            sequence.append("I")
            i = i - 1

            # print(sequence[::-1])
    return sequence[::-1]

def levenshtein(s1, s2):
    # if len(s1) < len(s2):
    #	return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    # print s1
    # print s2
    previous_row = range(len(s2) + 1)  # [0]*(len(s2)+1)#
    mat_out = []
    cost_mat = []
    cost_mat.append(previous_row)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        mat_in = []
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row
        # mat_out.append(mat_in)
        cost_mat.append(previous_row)
    # print(insertions)
    # print(deletions)
    # print(substitutions)
    # print(mat_out)
    # print(cost_mat)
    num_rows = len(cost_mat)
    num_col = len(cost_mat[num_rows - 1])
    # print cost_mat
    sequence = sequence_derive(cost_mat, num_rows, num_col)
    # pdb.set_trace()
    # print(sequence)
    return previous_row[-1], sequence
