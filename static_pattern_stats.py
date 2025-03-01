import dash
from dash import html, dcc, Output, Input, State, ctx
import dash_cytoscape as cyto
import pandas as pd
import numpy as np
import time

CANVAS_HEIGHT = '70vh'
SEGMENT_DELTA = '5vh'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df = pd.read_parquet('segment_data.parquet')
initial_elements = []

cyto_stylesheet = [
    {'selector': 'node', 'style': {
        'background-color': '#3498DB', 
        'width': '2.5vw', 
        'height': '2.5vw',
        'label': 'data(label)', 
        'color': '#2C3E50', 
        'font-size': '2vw',
        'text-valign': 'center', 
        'text-halign': 'right'
    }},
    {'selector': 'edge', 'style': {
        'line-color': '#7F8C8D', 
        'width': '0.4vw', 
        'curve-style': 'bezier'
    }},
    {'selector': 'edge.last-edge', 'style': {
        'line-color': '#27AE60', 
        'width': '0.6vw'
    }},
    {'selector': '.high-line', 'style': {
        'line-color': '#2C3E50', 
        'width': '0.2vw', 
        'target-arrow-shape': 'none', 
        'line-style': 'dashed'
    }},
    {'selector': '.low-line', 'style': {
        'line-color': '#2C3E50', 
        'width': '0.2vw', 
        'target-arrow-shape': 'none', 
        'line-style': 'dashed'
    }},
    {'selector': '.likely-high', 'style': {
        'line-color': '#2C3E50', 
        'width': '0.2vw', 
        'target-arrow-shape': 'none', 
        'line-style': 'dashed'
    }},
    {'selector': '.likely-low', 'style': {
        'line-color': '#2C3E50', 
        'width': '0.2vw', 
        'target-arrow-shape': 'none', 
        'line-style': 'dashed'
    }}
]

app.layout = html.Div([
    html.Div([
        # Header with logo and title
        html.Div([
            html.Img(src="https://via.placeholder.com/30", style={
                'height': '4vw',
                'maxHeight': '30px',
                'verticalAlign': 'middle',
                'marginRight': '1vw'
            }),
            html.H1("Market Metriks", style={
                'color': '#2C3E50',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': 'clamp(20px, 4vw, 28px)',
                'margin': '0'
            })
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center',
            'padding': '1vh 0',
            'flexWrap': 'wrap'
        }),
        
        html.P("statistics behind market structure patterns", style={
            'textAlign': 'center',
            'color': '#7F8C8D',
            'fontSize': 'clamp(12px, 2.5vw, 14px)',
            'margin': '1vh 0'
        }),

        # Button row
        html.Div([
            html.Button("Add Segment", id="add-btn", n_clicks=0, style={
                'backgroundColor': '#3498DB',
                'color': 'white',
                'border': 'none',
                'padding': 'clamp(6px, 1.5vw, 10px) clamp(12px, 3vw, 20px)',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontSize': 'clamp(12px, 2vw, 16px)',
                'margin': '0.5vw',
                'flex': '1 1 auto'
            }),
            html.Button("Remove Segment", id="remove-btn", n_clicks=0, style={
                'backgroundColor': '#E74C3C',
                'color': 'white',
                'border': 'none',
                'padding': 'clamp(6px, 1.5vw, 10px) clamp(12px, 3vw, 20px)',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontSize': 'clamp(12px, 2vw, 16px)',
                'margin': '0.5vw',
                'flex': '1 1 auto'
            }),
            html.Button("Search", id="search-btn", n_clicks=0, style={
                'backgroundColor': '#27AE60',
                'color': 'white',
                'border': 'none',
                'padding': 'clamp(6px, 1.5vw, 10px) clamp(12px, 3vw, 20px)',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontSize': 'clamp(12px, 2vw, 16px)',
                'margin': '0.5vw',
                'flex': '1 1 auto'
            }),
            html.Button("Reset", id="reset-btn", n_clicks=0, style={
                'backgroundColor': '#7F8C8D',
                'color': 'white',
                'border': 'none',
                'padding': 'clamp(6px, 1.5vw, 10px) clamp(12px, 3vw, 20px)',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontSize': 'clamp(12px, 2vw, 16px)',
                'margin': '0.5vw',
                'flex': '1 1 auto'
            }),
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'justifyContent': 'center',
            'gap': '1vw',
            'padding': '1vh 0',
            'maxWidth': '100%'
        }),

        # Loading and results
        dcc.Loading(
            id="loading-search",
            type="circle",
            children=html.Div(id="search-progress", style={
                'color': '#7F8C8D',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': 'clamp(12px, 2.5vw, 14px)',
                'textAlign': 'center',
                'margin': '1vh 0'
            })
        ),
        
        html.Div(id="search-result", style={
            'padding': '2vw',
            'backgroundColor': '#ECF0F1',
            'borderRadius': '5px',
            'fontFamily': 'Arial, sans-serif',
            'color': '#2C3E50',
            'fontSize': 'clamp(12px, 2.5vw, 14px)',
            'margin': '1vh 0',
            'width': '100%'
        }),

        # Code output
        html.Pre(id="code-output", style={
            'border': '1px solid #D5D8DC',
            'padding': '2vw',
            'backgroundColor': '#F9F9F9',
            'borderRadius': '5px',
            'fontSize': 'clamp(12px, 2vw, 14px)',
            'color': '#2C3E50',
            'overflowX': 'auto',
            'margin': '1vh 0',
            'width': '100%'
        }),

        # Cytoscape graph
        cyto.Cytoscape(
            id='cytoscape',
            elements=initial_elements,
            stylesheet=cyto_stylesheet,
            layout={'name': 'preset'},
            style={
                'width': '100%',
                'height': CANVAS_HEIGHT,
                'border': '1px solid #D5D8DC',
                'borderRadius': '5px',
                'backgroundColor': '#FFFFFF',
                'margin': '1vh 0'
            },
            responsive=True
        ),
        
        html.Div(id="search-state", style={'display': 'none'}, children="False")
    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'width': '90vw',
        'maxWidth': '1200px',
        'minWidth': '300px',
        'margin': '2vh auto',
        'padding': '2vw',
        'backgroundColor': '#F4F6F6',
        'borderRadius': '10px',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
    }),
    
    html.Footer([
        html.P("Powered by xAI", style={
            'textAlign': 'center',
            'color': '#7F8C8D',
            'fontSize': 'clamp(10px, 2vw, 12px)',
            'margin': '1vh 0'
        })
    ], style={
        'width': '100%',
        'marginTop': 'auto'
    })
], style={
    'display': 'flex',
    'flexDirection': 'column',
    'minHeight': '100vh',
    'padding': '0 2vw'
})

@app.callback(
    [Output('cytoscape', 'elements'),
     Output('search-progress', 'children'),
     Output('search-result', 'children'),
     Output('add-btn', 'disabled'),
     Output('remove-btn', 'disabled'),
     Output('search-btn', 'disabled'),
     Output('reset-btn', 'disabled'),
     Output('search-state', 'children')],
    [Input('add-btn', 'n_clicks'),
     Input('remove-btn', 'n_clicks'),
     Input('search-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks')],
    [State('cytoscape', 'elements'),
     State('search-state', 'children')],
    prevent_initial_call=True
)
def update_chain(n_add, n_remove, n_search, n_reset, elements, search_state):
    trigger_id = ctx.triggered_id
    if elements is None:
        elements = []
    nodes = [ele for ele in elements if 'source' not in ele.get('data', {}) and 'chainIndex' in ele.get('data', {})]
    search_performed = search_state == "True"

    if trigger_id == 'reset-btn':
        return [], "", "", False, False, False, False, "False"

    if trigger_id in ['add-btn', 'remove-btn']:
        if search_performed:
            for ele in elements:
                if 'position' in ele:
                    ele['grabbable'] = False
            return elements, dash.no_update, dash.no_update, True, True, True, False, "True"
        
        if trigger_id == 'add-btn':
            if not nodes:
                node0 = {'data': {'id': 'node-0', 'chainIndex': 0, 'label': 'Start'}, 'position': {'x': 20, 'y': 300}, 'grabbable': True}
                node1 = {'data': {'id': 'node-1', 'chainIndex': 1, 'label': ''}, 'position': {'x': 100, 'y': 250}, 'grabbable': True}
                edge1 = {'data': {'id': 'edge-1', 'source': 'node-0', 'target': 'node-1'}, 'classes': 'last-edge'}
                elements = [node0, node1, edge1]
            else:
                for ele in elements:
                    if 'source' in ele.get('data', {}):
                        ele['classes'] = ele.get('classes', '').replace('last-edge', '').strip()
                last_node = max(nodes, key=lambda n: n['data']['chainIndex'])
                last_chain_index = last_node['data']['chainIndex']
                nodes_sorted = sorted(nodes, key=lambda n: n["data"]["chainIndex"])
                previous_direction = "up" if len(nodes_sorted) < 2 else "up" if nodes_sorted[-1]["position"]["y"] < nodes_sorted[-2]["position"]["y"] else "down"
                new_y = last_node["position"]["y"] + 50 if previous_direction == "up" else last_node["position"]["y"] - 50
                new_chain_index = last_chain_index + 1
                new_node = {'data': {'id': f'node-{new_chain_index}', 'chainIndex': new_chain_index, 'label': ''}, 'position': {'x': last_node['position']['x'] + 80, 'y': new_y}, 'grabbable': True}
                new_edge = {'data': {'id': f'edge-{new_chain_index}', 'source': last_node['data']['id'], 'target': new_node['data']['id']}, 'classes': 'last-edge'}
                elements.extend([new_node, new_edge])
        elif trigger_id == 'remove-btn':
            if len(nodes) <= 1:
                return elements, dash.no_update, dash.no_update, False, False, False, False, "False"
            last_node = max(nodes, key=lambda n: n['data']['chainIndex'])
            remove_node_id = last_node['data']['id']
            remove_chain_index = last_node['data']['chainIndex']
            elements = [ele for ele in elements if not (ele.get('data', {}).get('chainIndex') == remove_chain_index or ('target' in ele.get('data', {}) and ele['data']['target'] == remove_node_id))]
            new_nodes = [ele for ele in elements if 'source' not in ele.get('data', {}) and 'chainIndex' in ele.get('data', {})]
            if new_nodes:
                new_last_node = max(new_nodes, key=lambda n: n['data']['chainIndex'])
                new_last_node['data']['label'] = 'Start' if len(new_nodes) == 1 else ''
                for ele in elements:
                    if 'source' in ele.get('data', {}):
                        ele['classes'] = ele.get('classes', '').replace('last-edge', '').strip()
                    if 'source' in ele.get('data', {}) and ele['data']['target'] == new_last_node['data']['id']:
                        ele['classes'] = 'last-edge'
        elements = [ele for ele in elements if not any(c in ele.get('classes', '') for c in ['high-line', 'low-line', 'likely-high', 'likely-low'])]
        return elements, dash.no_update, dash.no_update, False, False, False, False, "False"

    elif trigger_id == 'search-btn':
        if len(nodes) < 2:
            return elements, "Not enough segments to search (need at least 1 finalized segment).", "", False, False, False, False, "False"

        for ele in elements:
            if 'position' in ele:
                ele['grabbable'] = False

        finalized_nodes = nodes[:-1]
        pattern_relations = []
        for i in range(len(finalized_nodes) - 1):
            start_y = 600 - finalized_nodes[i]["position"]["y"]
            end_y = 600 - finalized_nodes[i + 1]["position"]["y"]
            if i == 0:
                direction = "up" if end_y > start_y else "down" if end_y < start_y else "flat"
                pattern_relations.append({"direction": direction})
            else:
                prev_start = 600 - finalized_nodes[i - 1]["position"]["y"]
                prev_end = 600 - finalized_nodes[i]["position"]["y"]
                relation = {
                    "start_vs_prev_start": "above" if start_y > prev_start else "below" if start_y < prev_start else "equal",
                    "end_vs_prev_start": "above" if end_y > prev_start else "below" if end_y < prev_start else "equal",
                    "end_vs_prev_end": "above" if end_y > prev_end else "below" if end_y < prev_end else "equal"
                }
                pattern_relations.append(relation)

        window_size = len(pattern_relations)
        if window_size < 1:
            return elements, "No finalized segments to search.", "", False, False, False, False, "False"

        start_time = time.time()
        start_prices = df['start_price'].values
        end_prices = df['end_price'].values
        total_steps = len(df) - window_size

        matches = 0
        trade_above_count = 0
        trade_below_count = 0
        directions = np.where(end_prices[:-1] > start_prices[:-1], "up", 
                              np.where(end_prices[:-1] < start_prices[:-1], "down", "flat"))

        for i in range(total_steps):
            window_start = start_prices[i:i + window_size]
            window_end = end_prices[i:i + window_size]
            window_relations = []
            if i < len(directions):
                window_relations.append({"direction": directions[i]})
            for j in range(1, window_size):
                start_price = window_start[j]
                prev_start = window_start[j - 1]
                prev_end = window_end[j - 1]
                end_price = window_end[j]
                relation = {
                    "start_vs_prev_start": "above" if start_price > prev_start else "below" if start_price < prev_start else "equal",
                    "end_vs_prev_start": "above" if end_price > prev_start else "below" if end_price < prev_start else "equal",
                    "end_vs_prev_end": "above" if end_price > prev_end else "below" if end_price < prev_end else "equal"
                }
                window_relations.append(relation)

            if window_relations == pattern_relations:
                matches += 1
                if i + window_size < len(df):
                    next_end_price = end_prices[i + window_size]
                    last_segment_start = window_start[-1]
                    last_segment_end = window_end[-1]
                    if last_segment_start > last_segment_end:
                        if next_end_price > last_segment_start:
                            trade_above_count += 1
                        elif next_end_price < last_segment_start:
                            trade_below_count += 1
                    else:
                        if next_end_price < last_segment_start:
                            trade_below_count += 1
                        elif next_end_price > last_segment_start:
                            trade_above_count += 1

        search_time = time.time() - start_time
        total_trades = trade_above_count + trade_below_count
        if total_trades > 0:
            prob_above = (trade_above_count / total_trades) * 100
            prob_below = (trade_below_count / total_trades) * 100
            if prob_above + prob_below < 100:
                remaining = 100 - (prob_above + prob_below)
                prob_above += remaining / 2
                prob_below += remaining / 2
        else:
            prob_above = prob_below = 0

        last_start_node = finalized_nodes[-2]
        last_end_node = finalized_nodes[-1]
        start_x, start_y = last_start_node['position']['x'], last_start_node['position']['y']
        end_x, end_y = last_end_node['position']['x'], last_end_node['position']['y']
        segment_x_distance = end_x - start_x

        high_line_end_x = start_x + segment_x_distance
        low_line_end_x = end_x + segment_x_distance
        high_line_end = {'data': {'id': 'high-line-end'}, 'position': {'x': high_line_end_x, 'y': start_y}, 'grabbable': False}
        low_line_end = {'data': {'id': 'low-line-end'}, 'position': {'x': low_line_end_x, 'y': end_y}, 'grabbable': False}
        high_line = {'data': {'source': last_start_node['data']['id'], 'target': 'high-line-end'}, 'classes': 'high-line'}
        low_line = {'data': {'source': last_end_node['data']['id'], 'target': 'low-line-end'}, 'classes': 'low-line'}

        if prob_above > prob_below:
            high_line['classes'] = 'likely-high'
        else:
            low_line['classes'] = 'likely-low'
        elements.extend([high_line_end, high_line, low_line_end, low_line])

        progress = "Search complete (100%)"
        result = (f"Pattern found {matches} times in historical data.\n"
                  f"Probability of trading to the most recent high first: {prob_above:.2f}%\n"
                  f"Probability of trading to the most recent low first: {prob_below:.2f}%\n"
                  f"Search time: {search_time:.2f} seconds")
        return elements, progress, result, True, True, True, False, "True"

@app.callback(
    Output('code-output', 'children'),
    Input('cytoscape', 'elements')
)
def generate_code(elements):
    if not elements:
        return "No pattern created yet."
    nodes = sorted([ele for ele in elements if 'source' not in ele.get('data', {}) and 'chainIndex' in ele.get('data', {})], key=lambda n: n["data"]["chainIndex"])
    if len(nodes) < 3:
        return "Not enough finalized segments for comparison."
    finalized_nodes = nodes[:-1]
    if len(finalized_nodes) < 2:
        return "Not enough finalized segments for comparison."

    output_lines = ["Rule: Segments must alternate direction (up, down, up, down, ...).\n"]
    seg_dirs = []

    def compare_relation(cur, ref):
        return "above" if cur > ref else "below" if cur < ref else "equal"

    for i in range(1, len(finalized_nodes)):
        start_y = 600 - finalized_nodes[i-1]["position"]["y"]
        end_y = 600 - finalized_nodes[i]["position"]["y"]
        current_dir = "up" if end_y > start_y else "down" if end_y < start_y else "flat"
        seg_dirs.append(current_dir)
        if i == 1:
            line = f"Segment 1: {current_dir}"
            if current_dir == "flat":
                line += "  *** ZIGZAG RULE BROKEN: flat segment not allowed ***"
            output_lines.append(line)
        else:
            prev_start = 600 - finalized_nodes[i-2]["position"]["y"]
            prev_end = 600 - finalized_nodes[i-1]["position"]["y"]
            line = (f"Segment {i}: start is {compare_relation(start_y, prev_start)} previous start; "
                    f"end is {compare_relation(end_y, prev_start)} previous start, {compare_relation(end_y, prev_end)} previous end")
            if seg_dirs[-1] == seg_dirs[-2]:
                line += f"  *** ZIGZAG RULE BROKEN: consecutive segments are {seg_dirs[-1]} ***"
            output_lines.append(line)
    return "\n".join(output_lines)


if __name__ == '__main__':
    # app.run_server(debug=True, port = 8050)
    app.run_server(host="0.0.0.0", port=8080)


