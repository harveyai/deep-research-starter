import logging
from typing import Iterable
from openai import OpenAI
import streamlit as st
import os
import enum
from openai.types.responses import (
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseStreamEvent,
    ResponseReasoningSummaryDeltaEvent,
    ResponseReasoningSummaryDoneEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
)
from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall,
)
from openai.types.responses.response_computer_tool_call import ResponseComputerToolCall
from openai.types.responses.response_file_search_tool_call import (
    ResponseFileSearchToolCall,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_web_search import (
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_item import (
    ImageGenerationCall,
    LocalShellCall,
    McpApprovalRequest,
    McpCall,
    McpListTools,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem


DEFAULT_SYSTEM_MESSAGE = """
You are an expert research assistant specializing in comprehensive analysis and synthesis of complex topics. Your role is to:
1. Conduct thorough research across multiple sources and perspectives
2. Analyze information critically and identify key insights
3. Synthesize findings into coherent, well-structured responses
4. Provide citations and evidence to support your conclusions
5. Highlight areas of uncertainty or conflicting information
Always strive for accuracy, objectivity, and depth in your research and analysis.
""".strip()

DEFAULT_USER_MESSAGE = (
    'Please conduct a comprehensive analysis of the FTC\'s "Click to Cancel" Rule.'
)


class DeepResearchModel(enum.Enum):
    O3_DEEP_RESEARCH = 'o3-deep-research-2025-06-26'
    O4_MINI_DEEP_RESEARCH = 'o4-mini-deep-research-2025-06-26'


DEFAULT_SYSTEM_MESSAGE = """
You are an expert research assistant specializing in comprehensive analysis and synthesis of complex topics. Your role is to:
1. Conduct thorough research across multiple sources and perspectives
2. Analyze information critically and identify key insights
3. Synthesize findings into coherent, well-structured responses
4. Provide citations and evidence to support your conclusions
5. Highlight areas of uncertainty or conflicting information
Always strive for accuracy, objectivity, and depth in your research and analysis.
""".strip()

DEFAULT_USER_MESSAGE = (
    'Please conduct a comprehensive analysis of the FTC\'s "Click to Cancel" Rule.'
)


def stream_deep_research_to_streamlit(stream: Iterable[ResponseStreamEvent]):
    """Stream deep research results to Streamlit UI components"""

    progress_container = st.container()
    reasoning_container = st.container()
    answer_container = st.container()

    with progress_container:
        st.subheader('Current Activity', divider=True)
        progress_placeholder = st.empty()

    with reasoning_container:
        st.subheader('Research Process', divider=True)
        reasoning_placeholder = st.empty()

    with answer_container:
        st.subheader('Research Results', divider=True)
        answer_placeholder = st.empty()

    reasoning_log = []
    current_activity = ''
    full_answer = ''

    for event in stream:
        # Handle reasoning summary text completion
        if isinstance(event, ResponseReasoningSummaryTextDoneEvent):
            reasoning_log.append(f'**Reasoning:** {event.text}')
            with reasoning_placeholder.container():
                for i, log_entry in enumerate(reasoning_log):
                    st.markdown(f'{i+1}. {log_entry}')

        # Handle text delta events (main answer content)
        elif isinstance(event, ResponseTextDeltaEvent):
            full_answer += event.delta
            answer_placeholder.markdown(full_answer)

        # Handle response creation and completion
        elif isinstance(event, ResponseCreatedEvent):
            full_answer += event.response.output_text
            answer_placeholder.markdown(full_answer)

        elif isinstance(event, ResponseCompletedEvent):
            full_answer = event.response.output_text
            answer_placeholder.markdown(full_answer)
            progress_placeholder.success('✅ Research completed!')

        # Handle tool calls and web search activities
        elif isinstance(
            event, (ResponseOutputItemDoneEvent, ResponseOutputItemAddedEvent)
        ):
            if isinstance(event.item, ResponseFunctionWebSearch):
                try:
                    item_dict = event.item.to_dict()
                    if 'action' in item_dict and isinstance(item_dict['action'], dict):
                        action_dict = item_dict['action']
                        action_type = action_dict.get('type')

                        if action_type == 'search':
                            query = action_dict.get('query', '')
                            current_activity = f'**Search:** {query}'
                            reasoning_log.append(current_activity)

                        elif action_type == 'open_page':
                            url = action_dict.get('url', '')
                            if url:
                                current_activity = f'**Reading page:** {url}'
                                reasoning_log.append(current_activity)

                        elif action_type == 'find_in_page':
                            search_pattern = action_dict.get('pattern', '').strip()
                            url = action_dict.get('url', '')
                            if url:
                                if search_pattern:
                                    current_activity = (
                                        f'**Page search:** "{search_pattern}" on {url}'
                                    )
                                    reasoning_log.append(current_activity)
                                else:
                                    current_activity = f'**Reading page:** {url}'
                                    reasoning_log.append(current_activity)

                        # Update UI with current activity and reasoning log
                        progress_placeholder.info(current_activity)
                        with reasoning_placeholder.container():
                            for i, log_entry in enumerate(reasoning_log):
                                st.markdown(f'{i+1}. {log_entry}')

                except Exception as e:
                    logging.error(f'Error processing web search event: {e}')

            # Handle other tool call types
            elif isinstance(
                event.item,
                (
                    ResponseOutputMessage,
                    ResponseFileSearchToolCall,
                    ResponseFunctionToolCall,
                    ResponseComputerToolCall,
                    ResponseReasoningItem,
                    ImageGenerationCall,
                    ResponseCodeInterpreterToolCall,
                    LocalShellCall,
                    McpCall,
                    McpListTools,
                    McpApprovalRequest,
                ),
            ):
                # These are handled but don't need special UI updates
                pass
            else:
                st.info(f'Unhandled item type: {event.item.type}')

        # Handle other event types (mostly pass-through)
        elif isinstance(
            event,
            (
                ResponseOutputItemDoneEvent,
                ResponseContentPartDoneEvent,
                ResponseInProgressEvent,
                ResponseTextDoneEvent,
                ResponseContentPartAddedEvent,
                ResponseAudioDeltaEvent,
                ResponseAudioDoneEvent,
                ResponseAudioTranscriptDeltaEvent,
                ResponseAudioTranscriptDoneEvent,
                ResponseReasoningSummaryTextDeltaEvent,
                ResponseReasoningSummaryDoneEvent,
                ResponseReasoningSummaryDeltaEvent,
                ResponseWebSearchCallInProgressEvent,
                ResponseWebSearchCallSearchingEvent,
                ResponseWebSearchCallCompletedEvent,
                ResponseReasoningSummaryPartAddedEvent,
                ResponseReasoningSummaryPartDoneEvent,
            ),
        ):
            # These events are handled implicitly or don't need special processing
            pass
        else:
            logging.warning(f'Unexpected event type: {type(event)}')


def main():
    st.title('Deep Research Sample')
    api_key = st.text_input(
        'OpenAI API Key (inherits from OPENAI_API_KEY env var by default)',
        type='password',
        value=os.getenv('OPENAI_API_KEY', ''),
    )
    model = st.selectbox('Model', options=list(DeepResearchModel), index=0)
    system_message = st.text_area(
        'System Message',
        value=DEFAULT_SYSTEM_MESSAGE,
        height=200,
        help="Define the AI's role and behavior for the research task",
    )

    user_message = st.text_area(
        'User Message',
        value=DEFAULT_USER_MESSAGE,
        height=100,
        help='Enter your research question or topic to analyze',
    )

    if not api_key:
        st.warning('OpenAI API Key is required')
        return
    if not st.button('Run'):
        return

    client = OpenAI(api_key=api_key)
    response_stream = client.responses.create(
        model=model.value,
        input=[
            {
                'role': 'developer',
                'content': [
                    {
                        'type': 'input_text',
                        'text': system_message,
                    }
                ],
            },
            {
                'role': 'user',
                'content': [{'type': 'input_text', 'text': user_message}],
            },
        ],
        reasoning={'summary': 'auto'},
        tools=[
            {'type': 'web_search_preview'},
            # Optional
            # {"type": "code_interpreter", "container": {"type": "auto", "file_ids": []}},
        ],
        stream=True,
    )

    with st.spinner('Running Deep Research...'):
        stream_deep_research_to_streamlit(response_stream)


if __name__ == '__main__':
    main()
