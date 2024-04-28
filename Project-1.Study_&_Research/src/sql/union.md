## UNION operator

The UNION operator is used to combine the result-set of two or more SELECT statements.

# Example

```sql
SELECT * FROM table1
UNION
SELECT * FROM table2;
```

e,g,1

```sql
SELECT first_name FROM employees
UNION
SELECT barach FROM branches;
```
