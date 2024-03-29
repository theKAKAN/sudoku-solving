# Checks whether it will be legal to assign num to the given row, col
def isPossibleToAssign(grid, row, col, num):
 
    # Check for same num in row
    for x in range(9):
        if grid[row][x] == num:
            return False
 
    # Check for same num in column
    for x in range(9):
        if grid[x][col] == num:
            return False
 
    # Check for same num in 3*3 matrix
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True

def solveSudokuPuzzle(grid, row, col):
    # 9x9 matrix
    N = 9
    # Check if we reached the end
    if (row == N - 1 and col == N):
        return True
 
    # Check if column value becomes 9, then move to the next row
    if col == N:
        row += 1
        col = 0
 
    # If it already has a value, move on
    if grid[row][col] > 0:
        return solveSudokuPuzzle(grid, row, col + 1)
    for num in range(1, N + 1, 1):
 
        # Check if it is safe to place the number at that spot
        if isPossibleToAssign(grid, row, col, num):
            grid[row][col] = num
 
            # Checking for next possibility with next col
            if solveSudokuPuzzle(grid, row, col + 1):
                return True
 
        # Removing the assigned num since our assumption was wrong
        grid[row][col] = 0
    return False