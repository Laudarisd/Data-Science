## WILDCARD

<p> A wildcard character is used to substitute one or more characters in a string.
Wildcard characters are used with the `LIKE` operator. The LIKE operator is used in a `WHERE` clause to search for a specified pattern in a column.</p>

=========================================================================

### Wild card Characters in MySQL

<p>The wildcard characters are:</p>

<ul>
<li>`%`: Matches zero or more characters</li>
<li>`_`: Matches exactly one character</li>
</ul>

=========================================================================

### LIKE Operator in MySQL

<p>The LIKE operator is used in a `WHERE` clause to search for a specified pattern in a column.</p>

e.g.1

```sql
SELECT * FROM table 
WHERE column LIKE '%pattern%';
```

=========================================================================

### Using the % wildcard character

e.g.1

```sql
SELECT * FROM Customers
WHERE City LIKE 'ber%';
```

The above query will return all the customers whose city starts with the letter 'ber'.


e.g.2

```sql
SELECT * FROM Customers
WHERE City LIKE '%es%';
```
The above query will return all the customers whose city containg the pattern 'es'.

=========================================================================

### Using the _ Wildcard Character

e.g.1 

The following SQL statement selects all customers with a City starting with any character, followed by "ondon":

```sql
SELECT * FROM Customers
WHERE City LIKE '_ondon';
```


e.g.2

The following SQL statement selects all customers with a City starting with "L", followed by any character, followed by "n", followed by any character, followed by "on":

```sql
SELECT * FROM Customers
WHERE City LIKE 'L_n_on';
```

