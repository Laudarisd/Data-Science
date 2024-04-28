## Link local folder to db
## Start db ``mongo``

we can see the following lines in terminal:

![$ mongo](1.png)

===============================================

Let's start

**Check database**

```
> show dbs
```
- Output should be 

```
admin   0.000GB
config  0.000GB
local   0.000GB
test    0.000GB

```

**Connect to brand new database**

Any database that we want

```
> use shop
# output
switched to db shop

```

**Insert one**
```
> db.products.insertOne({name:"WinterMouse", price:200,  Description:{detail:"Programming Language list"}})
{
	"acknowledged" : true,
	"insertedId" : ObjectId("615cc13d27cc5a493dee95d0") #db gives us an unique insertion id
}
```

**Check insertated data**

```
> db.products.find().pretty()
{
	"_id" : ObjectId("615cc19f27cc5a493dee95d1"),
	"name" : "WinterMouse",
	"price" : 200,
	"Description" : {
		"detail" : "Programming Language list"
	}
}
```
- Add more data

```
> db.products.insertOne({name:"WinterMouse", price:200,  Description:{detail:"Programming Language list", details:{available:"Online"}}})
{
	"acknowledged" : true,
	"insertedId" : ObjectId("615cc22027cc5a493dee95d2")
}
> db.products.find().pretty()
{
	"_id" : ObjectId("615cc19f27cc5a493dee95d1"),
	"name" : "WinterMouse",
	"price" : 200,
	"Description" : {
		"detail" : "Programming Language list"
	}
}
{
	"_id" : ObjectId("615cc22027cc5a493dee95d2"),
	"name" : "WinterMouse",
	"price" : 200,
	"Description" : {
		"detail" : "Programming Language list",
		"details" : {
			"available" : "Online"
		}
	}
}
```



