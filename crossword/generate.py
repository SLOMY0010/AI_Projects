import sys

from crossword import Variable, Crossword

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word. 
        """
        # Variables are keys and sets of invalid words (don't satisfy the variable's unary constraints) are values.
        words_to_remove = {
            var : set() for var in self.crossword.variables 
        }

        # Find the invalid words.
        for v in self.crossword.variables:
            for word in self.domains[v]:
                if len(word) != v.length:
                    words_to_remove[v].add(word)

        # Remove the invalid words from the domains of each variable.
        for key in words_to_remove:
            for value in words_to_remove[key]:
                self.domains[key].remove(value)


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        revised = False

        # A set to store valid words.
        valid_words = set()
        
        # Find the valid words and store them in the valid_words set.
        i, j = self.crossword.overlaps[x, y]
        for y_word in self.domains[y]:
            for x_word in self.domains[x]:           
                if x_word != y_word and x_word[i] == y_word[j]:
                    valid_words.add(x_word)
                    revised = True

        # Set the domain of the variable to be a set of the valid words only.
        self.domains[x] = valid_words

        return revised


    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # If arcs is None, set arcs to be a list of all arcs in the problem.
        if arcs == None:
            arcs = list()
            for x in self.crossword.variables:
                for y in self.crossword.neighbors(x):
                    arcs.append((x, y))

        # Check the consistency of each arc.
        while len(arcs) > 0:
            x, y = arcs.pop()
            if self.revise(x, y):

                # If x's domain if empty, then the problem is unsolvable.
                if self.domains[x] == 0:
                    return False

                # If a change was made to x'd domain and it is not empty
                # add other pairs of x and its neighbors except y to the queue, to check their consistency.
                for z in self.crossword.neighbors(x) - {y}:
                    arcs.insert(0, (z, x))

        return True


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # If not every single variable was in the assignment, then the assignment is not complete.
        for var in self.crossword.variables:
            if var not in assignment:
                return False

        return True


    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Check for uniqueness.
        values = list(assignment.values())
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                if values[i] == values[j]:
                    return False

        # Check if every value has the correct length.
        for key, value in assignment.items():
            if key.length != len(value):
                return False

        # Check for conflicts.
        arcs = list()
        for x in self.crossword.variables:
            for y in self.crossword.neighbors(x):
                arcs.append((x, y))

        for x in assignment:
            for y in assignment:
                if x != y and y in self.crossword.neighbors(x):
                    i, j = self.crossword.overlaps[x, y]
                    if assignment[x][i] != assignment[y][j]:
                            return False

        # If the assignment passed all of the three tests, return True.     
        return True


    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        domain_values = {
            word : 0 for word in self.domains[var]
        }

        for word in domain_values:
            for neighbor in self.crossword.neighbors(var):
                if word in self.domains[neighbor] and neighbor not in assignment:
                    domain_values[word] += 1

        domain_values = dict(sorted(domain_values.items(), key=lambda x:x[1]))

        return list(domain_values.keys())


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # The highest number of neighbors.
        max_degree = max([len(self.crossword.neighbors(x)) for x in self.crossword.variables if x not in assignment])
        
        # The lowest number of mrv.
        mrv_vars = self.find_mrv(assignment)

        if len(mrv_vars) > 1:
            for var in mrv_vars:
                if len(self.crossword.neighbors(var)) == max_degree:
                    return var

        return mrv_vars[0]


    def find_mrv(self, assignment):
        """
        Returns a list of variables that have the minimum remaining value that are not assigned yet.
        """
        min_domain = min([len(self.domains[x]) for x in self.crossword.variables if x not in assignment])
        mrv_vars = list()

        for var in self.crossword.variables:
            if var not in assignment and len(self.domains[var]) == min_domain:
                mrv_vars.append(var)
        return mrv_vars


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        
        var = self.select_unassigned_variable(assignment)
        for value in self.domains[var]:
            assignment[var] = value
            if not self.consistent(assignment):
                del assignment[var]
                continue
            result = self.backtrack(assignment)
            if result != None:
                return result
            del assignment[var]
        
        return None
            


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words) 
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()