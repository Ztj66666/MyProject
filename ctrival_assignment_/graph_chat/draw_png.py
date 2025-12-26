from graph_chat.log_utils import log


def draw_graph(graph,file_name):
    try:
        mermaid_node = graph.get_graph().draw_mermaid_png()
        with open(file_name,"wb") as f:
            f.write(mermaid_node)
    except Exception as e:
        log.exception(e)

