curl -X POST -H "Content-Type: application/json" -d '{
  "case_id": "1",
  "uri": "https://elephantscale-public.s3.amazonaws.com/data/text/books.zip "
}' http://localhost:5000/cases

