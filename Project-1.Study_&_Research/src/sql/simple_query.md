# SIMPLE MYSQL COMMANDS
--------------------------------------

Note: all the command are exucated in `popsql`. It can also be done in terminal.

To connect to popsql:

```
hostname: ....
port:...
database: Table name
```
=====================================================================

### CREATE TABLE

```sql
CREATE TABLE student;
```
will create a table name student.


For more information while creating table, we can write in this way:

```sql
CREATE TABLE student(
    student_id INT PRIMARY KEY,
    name VARCHAR(20),
    major VARCHAR(20)
);
```
In another way, we can setup PRIMARY KEY in this way

```sql
CREATE TABLE student PRIMARY KEY student_id;
```

=====================================================================

### CHECK CREATED TABLE

```sql
DESCRIBE student
```

This will show created data

=====================================================================

### RENAME TABLE

```sql
ALTER TABLE student
RENAME TO new_table_name;
```

=====================================================================

### ADD EXTRA COLUMN IN CREATED DATA

```sql
ALTER TABLE studen ADD gpa DECIMAL(3,2);
```

***Add multiple columns***
```sql
ALTER TABLE table_name
    ADD new_column_name column_definition
    [FIRST | AFTER column_name],
    ADD new_column_name column_definition
    [FIRST | AFTER column_name],
    ...;
Code language: SQL (Structured Query Language) (sql)
```

More examples:
```sql
ALTER TABLE vehicles
ADD color VARCHAR(50),
ADD note VARCHAR(255);
```
=====================================================================

### MODIFY COLUMN

```sql
ALTER TABLE table_name
MODIFY column_name column_definition
[ FIRST | AFTER column_name];    
```

If we want to change column name a NOT NULL then we can do the following:
```sql
ALTER TABLE student 
MODIFY name VARCHAR(20) NOT NULL;
```


***Modify multiple columns***
```sql
ALTER TABLE table_name
    MODIFY column_name column_definition
    [ FIRST | AFTER column_name],
    MODIFY column_name column_definition
    [ FIRST | AFTER column_name],
    ...;
```

e.g. we have a table name `vehicles`. We want to modify column `year` and `color` then we can do the following:
```sql
ALTER TABLE vehicles 
MODIFY year SMALLINT NOT NULL,
MODIFY color VARCHAR(20) NULL AFTER make;
```
In this example:

First, modify the data type of the year column from INT to SMALLINT
Second, modify the color column by setting the maximum length to 20, removing the NOT NULL constraint, and changing its position to appear after the make column.

=====================================================================

### DELETE ENTIRE DATA

```sql
DELETE TABLE student;
```

This will delete entire dataset


***Delete specific column***

```sql
ALTER TABLE student DROP COLUMN gpa;
```
This will drop `gpa` column from our table.


=====================================================================

### INSERT DATA 

```sql
INSERT INTO student VALUES (1, 'JACK', 'Biology')
```

This command will insert 3 values in student table. 

***If we don't know attribute***

```sql
INSERT INTO student (student_id, name) VALUES (1, 'Jack')
```

This comand will insert two values. Since we didn't know remaining attributes, it will automatically insert `NULL` in remaining attributes.

------------------------------------------------------
Let's define new attribute

```sql
ALTER TABLE studen ADD major VARCHAR(20) UNIQUE;
```

This command will add new attribute in student table. However, we have a rules for entries. We can't insert similar major and lenght is 20 characters.


***If it is given NO NULL***

In attribute insert criteria, if it is given `NO NULL` then there shouldn't be emty entry while inserting entries


***Default criteria***

If we write `DEFAULT 'Undecided' instead of UNIQUE then this will insert `undecided` if the entries are empty

=====================================================================

### AUTO INCREMENT

```sql
CREATE TABLE student (
    student_id int NOT NULL AUTO_INCREMENT,
    LastName varchar(20) NOT NULL,
    FirstName varchar(20),
    major CARCHAR(20) NOT NULL,
    gps DECIMALS(3, 2)
    PRIMARY KEY (student_id)
);
```

By default, the starting value for AUTO_INCREMENT is 1, and it will increment by 1 for each new record

=====================================================================

### UPDATE

To update date table 'student'

```sql
SELECT * FROM student;
UPDATE student
SET 
major = 'Computer'
WHERE student_id = 4;
```

***Update more than one values ***
If we have table `employes`

```
SELECT * FROM employes
UPDATE employes
SET
lastname = 'Hill',
    email = 'mary.hill@classicmodelcars.com'
WHERE
    employeeNumber = 1056;
```
```sql
UPDATE employees
SET email = REPLACE(email,'@classicmodelcars.com','@mysqltutorial.org')
WHERE
   jobTitle = 'Sales Rep' AND
   officeCode = 6;
```


***Update with more condition***
```sql
SELECT * FROM student;
UPDATE student
SET 
major = 'Computer'
WHERE major = 'Bio' or major = 'Chem' ;
```

=====================================================================

### DELETE ROW

```sql
SELECT * FROM student
DELETE FROM student
WHERE student_id = 5;
```
We can also use `and` condition
```
WHERE name = 'TOM' AND major = 'Undecided';
```

=====================================================================

### ORDER BY

The ORDER BY keyword is used to sort the result-set in ascending or descending order.

```sql
SELECT * FROM table_name
ORDER BY  colume1, column2,.... ASC|DESC;
```

e.g. 
```sql
SELECT name FROM student

-- or we can do following

SELECT student.name, student.major
FROM student
ORDER BY student_id ASC|DESC;
```

e.g. 2

```sql
SELECT * FROM Customers
ORDER BY Country DESC;
```
=====================================================================

### LIMIT

```sql
SELECT 
    select_list
FROM
    table_name
LIMIT [offset,] row_count;
```

e.g.1
```sql
SELECT 
    customerNumber, 
    customerName, 
    creditLimit
FROM
    customers
ORDER BY creditLimit DESC
LIMIT 5;
```
In this example:

- First, the ORDER BY clause sorts the customers by credits in high to low.
- Then, the LIMIT clause returns the first 5 rows


=====================================================================

### COUNT , AVERAGE , MAX , MIN , SUM

***COUNT***
```sql
SELECT COUNT(column_name) 
FROM table_name;

```

e.g.1
```sql
SELECT COUNT(emp_id)
FROM employees;
WHERE sex = 'F' AND birth-date > 1970-01-01;
```

e.g.2 
We can also use `count` with `GROUP BY`

```sql
SELECT product, COUNT(*)
FROM products
GROUP BY productline;
```

e.g.3 

To find how many males and females are there are

```sql
SELECT COUNT(sex), sex
FROM employee
GROUP BY sex;
``` 


***AVE***

Let's find out average salary of male from employees table

e.g.1

```sql
SELECT AVG(salary)
FROM employees;
WHERE sex = 'M';
```
e.g.2

```sql
SELECT 
    AVG(buyprice) 'Average Classic Cars Price'
FROM
    products
WHERE
    productline = 'Classic Cars';
```

***MAX***
```sql
SELECT 
    MAX(amount) largest_payment_2004
FROM
    payments
WHERE
    YEAR(paymentDate) = 2004;
```
In this example:

First, use a condition in the WHERE clause to get only payments whose year is 2004. We used the YEAR() function to extract year from the payment date.
Then, use the MAX() function in the SELECT clause to find the largest amount of payments in 2004.

e.g. 1

```sql
SELECT 
    *
FROM
    payments
WHERE
    amount = (SELECT 
            MAX(amount)
        FROM
            payments);
```

e.g.2 

```sql
SELECT * FROM payments
WHERE amount = (SELECT MAX(amount) 
                FROM payments);
```

***MIN***

e.g. 1

```sql
SELECT 
    MIN(buyPrice)
FROM
    products
WHERE
    productline = 'Motorcycles';
```

e.g. 2


```sql
SELECT 
    productCode, 
    productName, 
    buyPrice
FROM
    products
WHERE
    buyPrice = (
        SELECT 
            MIN(buyPrice)
        FROM
            products);
```


***SUM***

The following shows the order line items of the order number 10110:
```sql
SELECT 
    orderNumber, 
    quantityOrdered, 
    priceEach
FROM
    orderdetails
WHERE
    orderNumber = 10100;
```
To calculate the total for the order number 10110, you use the SUM() function as follows:

```sql
SELECT 
	SUM(quantityOrdered * priceEach)  orderTotal
FROM
	orderdetails
WHERE
	orderNumber = 10100;

```

*** SUM with GROUP BY***

```sql
SELECT 
    orderNumber, 
    SUM(quantityOrdered * priceEach) orderTotal
FROM
    orderdetails
GROUP BY 
    orderNumber
ORDER BY 
    orderTotal DESC;
```

=====================================================================

### CASE

The following SQL goes through conditions and returns a value when the first condition is met:

e.g. 1


```sql
SELECT OrderID, Quantity,
CASE
    WHEN Quantity > 30 THEN 'The quantity is greater than 30'
    WHEN Quantity = 30 THEN 'The quantity is 30'
    ELSE 'The quantity is under 30'
END AS QuantityText
FROM OrderDetails;
```
e.g. 2



```sql
SELECT CustomerName, City, Country
FROM Customers
ORDER BY
(CASE
    WHEN City IS NULL THEN Country
    ELSE City
END);
```


=====================================================================


<li> <a href = "https://github.com/Laudarisd/Data-science-study/tree/master/src/sql/function.md"> Function in Query </a> </li>

=====================================================================




### References
<li> <a href = "https://www.mysqltutorial.org/mysql-update-data.aspx"> ref 1 </a> </li>
<li> <a href = "https://www.w3schools.com/sql/sql_dates.asp"> ref 2 </a> </li>
