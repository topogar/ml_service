# ml_service

## CURL /forward OK
curl -X POST http://localhost:9001/forward -d '{"message": "John go walk"}' -H "Content-Type: application/json" - ham
curl -X POST http://localhost:9001/forward -d '{"message": "URGENT! Your mobile was awarded a å£1,500 Bonus Caller Prize on 27/6/03. Our final attempt 2 contact U! Call 08714714011"}' -H "Content-Type: application/json" - spam
## CURL /forward 403
-

## CURL /forward 400
curl -X POST http://localhost:9001/forward -d '{"messssage": None}' -H "Content-Type: application/json"