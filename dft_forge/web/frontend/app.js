import { initChat } from "./chat.js";
import { initViewer } from "./viewer.js";
import { initSettings } from "./settings.js";
import { initWorkspace } from "./workspace.js";

initChat();
initSettings();
initWorkspace();
setTimeout(initViewer, 0);
