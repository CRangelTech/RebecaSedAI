// Este archivo corre dentro del plugin de Figma
// code.js
figma.showUI(__html__, { width: 600, height: 500 });



figma.ui.onmessage = async (msg) => {
  if (msg.type === 'generate-ui') {
    const prompt = msg.prompt;

    const response = await fetch("http://localhost:5000/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });

    const result = await response.json();
    const layout = result.layout;

    const spacing = 20;
    const componentWidth = 300;
    const componentHeight = 50;
    let y = 0;

    const nodes = [];

    for (const item of layout) {
      let node;

      switch (item.type) {
        case "input":
          node = figma.createFrame();
          node.resize(componentWidth, componentHeight);
          node.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }];
          node.strokes = [{ type: "SOLID", color: { r: 0.7, g: 0.7, b: 0.7 } }];
          node.strokeWeight = 1;

          if (item.label) {
            const label = figma.createText();
            await figma.loadFontAsync({ family: "Roboto", style: "Regular" });
            label.characters = item.label;
            label.fontSize = 14;
            label.y = -20;
            node.appendChild(label);
          }

          break;

        case "button":
          node = figma.createFrame();
          node.resize(componentWidth, componentHeight);

          // Asignar color
          let color = { r: 0.3, g: 0.3, b: 0.3 }; // gris por default
          if (item.color === "blue") color = { r: 0.2, g: 0.4, b: 0.9 };
          if (item.color === "green") color = { r: 0.1, g: 0.6, b: 0.2 };
          if (item.color === "red") color = { r: 0.8, g: 0.2, b: 0.2 };
          if (item.color === "orange") color = { r: 1.0, g: 0.6, b: 0.1 };
          if (item.color === "gray") color = { r: 0.6, g: 0.6, b: 0.6 };

          node.fills = [{ type: "SOLID", color }];

          if (item.text) {
            const text = figma.createText();
            await figma.loadFontAsync({ family: "Roboto", style: "Bold" });
            text.characters = item.text;
            text.fontSize = 16;
            text.fills = [{ type: "SOLID", color: { r: 1, g: 1, b: 1 } }];
            text.x = 10;
            text.y = 12;
            node.appendChild(text);
          }

          break;

        case "text":
          node = figma.createText();
          await figma.loadFontAsync({ family: "Roboto", style: "Regular" });
          node.characters = item.content || "Texto";
          node.fontSize = 20;
          node.fills = [{ type: "SOLID", color: { r: 0, g: 0, b: 0 } }];
          break;

        case "image":
          node = figma.createFrame();
          node.resize(componentWidth, componentHeight);
          node.fills = [{ type: "SOLID", color: { r: 0.8, g: 0.8, b: 0.8 } }];
          if (item.description) {
            const desc = figma.createText();
            await figma.loadFontAsync({ family: "Roboto", style: "Italic" });
            desc.characters = item.description;
            desc.fontSize = 12;
            desc.y = 12;
            desc.x = 10;
            node.appendChild(desc);
          }
          break;

        default:
          node = figma.createRectangle();
          node.resize(componentWidth, componentHeight);
          node.fills = [{ type: "SOLID", color: { r: 0.5, g: 0.5, b: 0.5 } }];
      }

      node.x = 100;
      node.y = y;
      y += componentHeight + spacing;
      figma.currentPage.appendChild(node);
      nodes.push(node);
    }

    figma.viewport.scrollAndZoomIntoView(nodes);
  }
};
