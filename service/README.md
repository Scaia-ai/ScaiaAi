## Service


Currently the service writes data out to local pickle files.  This won't be very robust so we will likely switch to using sqllite.


![](../images/poc-ai.png)




## Running

To run the service, run the bash shell script:

```bash
cd service
bash ./run_service.sh
```

You will get a response something like this:

```console
 * Serving Flask app "api" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 814-782-702
```

## Testing 

In a separate window, type in the CURL:

```bash
curl http://localhost:5000
```

And you should get the response something like this:

```console
{ 'docsimilrity' : 'service'}
```


### API:

Cases:

* Do a POST to create a case

```console
curl -X POST -H "Content-Type: application/json" -d '{
  "case_id": "1",
  "uri": "https://elephantscale-public.s3.amazonaws.com/data/text/books.zip "
}' http://localhost:5000/cases

```

This will also trigger a download of all the files, unzip, etc.

* Do a GET to retrieve a case or cases

Get all cases: `curl http://localhost:5000/cases`   

Get one case by specifying the `case_id` : `curl http://loocalhost:5000/case/1`

* Do a GET to retrieve documents within a case

Get documents (pass `case_id`, `doc_id`): `curl http://localhost:5000/cases/1/docs/1`

* Do a GET to retrieve similar documetns:

Get documents (pass `case_id`, `doc_id`): `curl http://localhost:5000/cases/1/similar/1`



## Using Examples


### Adding a case (POST)

```console 

curl -X POST -H "Content-Type: application/json" -d '{
  "case_id": "1",
  "uri": "https://elephantscale-public.s3.amazonaws.com/data/text/books.zip "
}' http://localhost:5000/cases


```


### Getting a case (GET)

```console 

curl --request GET curl http://localhost:5000/cases/1 

```



### Generating Similar Documents (GET)

Here we pass the case ID, and the document ID

```console

curl --request GET curl http://localhost:5000/cases/1/similar/1

```
