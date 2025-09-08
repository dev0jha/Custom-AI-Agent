import { MemorySaver, MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import readlines from "node:readline/promises";
import { ChatGroq } from "@langchain/groq";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";

const checkpointer = new MemorySaver();

const tool = new TavilySearch({
  maxResults: 3,
  topic: "general",
  // customize tool options as needed
});

const tools = [];
const toolNode = new ToolNode(tools);

const llm = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0,
  maxTokens: undefined,
  maxRetries: 2,
}).bindTools(tool);

async function callModel(state) {
  console.log("Calling Model");
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
}

function shouldContinue(state) {
  const lastMessage = state.messages[state.messages.length - 1];
  console.log("state", state);
  if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
    return "tools";
  }
  return "__end__";
}

const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue);

const app = workflow.compile({ checkpointer });

async function main() {
  const rl = readlines.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  while (true) {
    const userInput = await rl.question("You: ");
    if (userInput === "exit") break;

    const finalState = await app.invoke(
      {
        messages: [{ role: "user", content: userInput }],
      },
      {
        configurable: {
          thread_id: "1", // Use snake_case here
        },
      }
    );

    const lastMessage = finalState.messages[finalState.messages.length - 1];
    console.log("AI:", lastMessage.content);
  }

  rl.close();
}

main();
