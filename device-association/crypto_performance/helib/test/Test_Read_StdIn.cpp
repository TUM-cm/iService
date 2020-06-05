#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iterator>

using namespace std;
#define BUFFER_SIZE BUFSIZ /* or some other number */

// g++ test.cpp
// ./a.out
// 1 2 3;4 5 6

int main(int argc, char **argv)
{
    vector<long> v1;
    vector<long> v2;

    char delimeter[] = ";";
    char x[BUFFER_SIZE];
    fgets(x, BUFSIZ, stdin);

    int input;
    while (strcmp(x, "quit\n") != 0) {

        char* token = strtok(x, delimeter);
        istringstream inv1(token);
        while(inv1 >> input) {
          v1.push_back(input);
        }

        token = strtok(NULL, delimeter);
        istringstream inv2(token);
        while(inv2 >> input) {
          v2.push_back(input);
        }

        for (vector<long>::const_iterator i = v1.begin(); i != v1.end(); ++i)
          cout << *i;
        cout << endl;

        for (vector<long>::const_iterator i = v2.begin(); i != v2.end(); ++i)
          cout << *i;
        cout << endl;

        fgets(x, BUFSIZ, stdin);
    }
}
