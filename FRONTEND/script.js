// Clothing sample data
const data = {
  hat: [
    { name: "Boné Colorido", img: "https://via.placeholder.com/260x260?text=Cap+1" },
    { name: "Gorro", img: "https://via.placeholder.com/260x260?text=Hat+2" }
  ],
  shirt: [
    { name: "Camisa Jeans", img: "https://via.placeholder.com/260x260?text=Shirt+1" },
    { name: "Camiseta Branca", img: "https://via.placeholder.com/260x260?text=Shirt+2" }
  ],
  pants: [
    { name: "Calça Jeans", img: "https://via.placeholder.com/260x260?text=Pants+1" },
    { name: "Bermuda", img: "https://via.placeholder.com/260x260?text=Pants+2" }
  ]
};

let index = { hat: 0, shirt: 0, pants: 0 };

const container = document.getElementById("carouselContainer");

function render() {
  container.innerHTML = "";
  
  Object.keys(data).forEach(key => {
    const item = data[key][index[key]];

    const block = document.createElement("div");
    block.className = "carousel-item";
    block.innerHTML = `
      <div class="arrow left" onclick="change('${key}', -1)">‹</div>
      <img src="${item.img}" alt="${item.name}">
      <p><b>${item.name}</b></p>
      <p class="mini-text">Sugestão: clima leve</p>
      <div class="arrow right" onclick="change('${key}', 1)">›</div>
    `;

    container.appendChild(block);
  });
}

function change(category, step) {
  const max = data[category].length;
  index[category] = (index[category] + step + max) % max;
  render();
}

render();

// Modal
const modal = document.getElementById("modalRegister");
document.getElementById("btnRegister").onclick = () => modal.classList.remove("hidden");
document.getElementById("closeRegister").onclick = () => modal.classList.add("hidden");