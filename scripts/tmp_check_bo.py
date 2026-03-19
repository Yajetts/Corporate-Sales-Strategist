import json
from src.api.app import create_app

app = create_app('testing')
app.config['TESTING'] = True

sample_product_portfolio = [
    {'name':'Product A','sales_history':[100,110,105],'demand_forecast':120,'production_cost':50,'current_inventory':20},
    {'name':'Product B','sales_history':[80,90,85],'demand_forecast':95,'production_cost':40,'current_inventory':10},
    {'name':'Product C','sales_history':[150,155,160],'demand_forecast':170,'production_cost':60,'current_inventory':30},
    {'name':'Product D','sales_history':[60,65,70],'demand_forecast':75,'production_cost':35,'current_inventory':15},
    {'name':'Product E','sales_history':[200,210,205],'demand_forecast':220,'production_cost':70,'current_inventory':25},
]

payload = {'product_portfolio': sample_product_portfolio, 'revenue_weight': 0.7, 'cost_weight': 0.3}

with app.test_client() as client:
    resp = client.post('/api/v1/business_optimizer', data=json.dumps(payload), content_type='application/json')
    print('status', resp.status_code)
    data = resp.get_json(silent=True)
    if isinstance(data, dict):
        print('keys', sorted(data.keys()))
        if 'error' in data:
            print('error', data.get('message'))
    else:
        print('non-dict response', type(data))
