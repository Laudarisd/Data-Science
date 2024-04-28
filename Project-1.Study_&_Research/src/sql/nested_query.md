 ## NESTED QUERIES
 
 It is like adding sub queries to the main query.

We do this by following query:

```sql
SELECT 
    lastName, firstName
FROM
    employees
WHERE
    officeCode IN (SELECT 
            officeCode
        FROM
            offices
        WHERE
            country = 'USA');
```


Let's see some examples:

e.g.1

```sql
-- Find names of all employes who have sold over 30,000 to a single client

SELECT employees.emp_id
FROM employees
WHERE employees.emp_id IN (SELECT emp_id
                            FROM sales
                            WHERE sales.amount > 30000);
```


e.g.2

```sql
-- Find all clients who are handles by the branch

SELECT clients.client_name
FROM clients
WHERE clients.client_id IN (SELECT client_id
                            FROM sales
                            WHERE sales.branch_id = 102);
```

=============================================================================

e.g.3

```sql
-- For example, we can find customers whose payments are greater than the average payment using a subquery:

SELECT 
    customerNumber, 
    checkNumber, 
    amount
FROM
    payments
WHERE
    amount > (SELECT 
            AVG(amount)
        FROM
            payments);
```

e.g.4

```sql
-- a subquery with NOT IN operator to find the customers who have not placed any orders as follows:
SELECT 
    customerName
FROM
    customers
WHERE
    customerNumber NOT IN (SELECT DISTINCT
            customerNumber
        FROM
            orders);
```


e.g.5

```sql
SELECT 
    MAX(items), 
    MIN(items), 
    FLOOR(AVG(items))
FROM
    (SELECT 
        orderNumber, COUNT(orderNumber) AS items
    FROM
        orderdetails
    GROUP BY orderNumber) AS lineitems;
```


e.g.6

```sql
-- The following example uses a correlated subquery to select products whose buy prices are greater than the average buy price of all products in each product line.

SELECT 
    productname, 
    buyprice
FROM
    products p1
WHERE
    buyprice > (SELECT 
            AVG(buyprice)
        FROM
            products
        WHERE
            productline = p1.productline)

```


e.g.7

```sql

SELECT
    customerNumber, customerName, country
FROM customers
WHERE
    EXISTS ( SELECT
            orderNumber, SUM(priceEach * quantityOrdered)
        FROM
            orderdeatails
            INNER JOIN
            orders USING (orderNumber)
        WHERE
            customers.customerNumber = orders.customerNumber
        GROUP BY orderNumber
        HAVING SUM(priceEach * quantityOrdered) > 60000);

```