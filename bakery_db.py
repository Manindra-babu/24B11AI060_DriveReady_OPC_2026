"""
Bakery Database and System Prompt Configuration for Sweet Crumbs Bakery Agent
"""

BAKERY_MENU = {
    "Cakes": [
        {"name": "Belgian Dark Chocolate Cake", "price": "$28 (500g) / $50 (1kg)", "dietary": "Egg & Eggless available", "desc": "Rich 70% dark Belgian ganache with moist sponge."},
        {"name": "Red Velvet Cream Cheese Cake", "price": "$30 (500g) / $55 (1kg)", "dietary": "Egg & Eggless available", "desc": "Traditional cocoa sponge layered with vanilla bean cream cheese frosting."},
        {"name": "Classic Vanilla Bean Cake", "price": "$24 (500g) / $45 (1kg)", "dietary": "100% Eggless", "desc": "Light sponge infused with pure Madagascar vanilla."},
        {"name": "Fresh Mango Mascarpone (Seasonal)", "price": "$32 (500g) / $58 (1kg)", "dietary": "Eggless available", "desc": "Fresh Alphonso mango slices with whipped mascarpone."}
    ],
    "Pastries & Desserts": [
        {"name": "Blueberry Baked Cheesecake Slice", "price": "$6.00", "dietary": "Contains Egg", "desc": "New York style rich cheesecake with blueberry compote."},
        {"name": "Dutch Truffle Pastry", "price": "$5.00", "dietary": "100% Eggless", "desc": "Layered chocolate truffle pastry slice."},
        {"name": "Choco Lava Molten Cupcake", "price": "$4.50", "dietary": "Eggless available", "desc": "Warm gooey molten chocolate center."},
        {"name": "Assorted French Macarons (Box of 6)", "price": "$14.00", "dietary": "Contains Egg (Gluten-Free)", "desc": "Pistachio, Raspberry, Salted Caramel, Chocolate, Vanilla, Lemon."}
    ],
    "Artisanal Breads": [
        {"name": "Classic Country Sourdough Loaf", "price": "$8.00", "dietary": "Vegan / 100% Eggless", "desc": "36-hour naturally fermented sourdough with crisp crust."},
        {"name": "Garlic Herb Rosemary Focaccia", "price": "$7.00", "dietary": "Vegetarian", "desc": "Extra virgin olive oil dough topped with roasted garlic and fresh herbs."},
        {"name": "Butter Croissant", "price": "$4.00", "dietary": "Contains Egg Wash & Dairy", "desc": "Flaky 27-layer French butter croissant."},
        {"name": "Seeded Multigrain Sandwich Bread", "price": "$6.50", "dietary": "Vegan / 100% Eggless", "desc": "Wholesome bread baked with flax, pumpkin, and sunflower seeds."}
    ],
    "Cookies & Savory": [
        {"name": "Double Choc-Chip Walnut Cookie", "price": "$3.50", "dietary": "Eggless available", "desc": "Crispy edges with a chewy chocolate center."},
        {"name": "Almond Pistachio Biscotti (Pack of 4)", "price": "$7.00", "dietary": "100% Eggless", "desc": "Twice-baked crunchy Italian biscuits, perfect with coffee."},
        {"name": "Paneer Spiced Puff Pastry", "price": "$4.50", "dietary": "Vegetarian", "desc": "Flaky golden crust stuffed with spiced cottage cheese filling."}
    ]
}

BAKERY_INFO = {
    "name": "Sweet Crumbs Bakery & Cafe",
    "hours": "Tuesday to Sunday, 9:00 AM – 8:00 PM (Closed on Mondays)",
    "phone": "+1-555-019-2253",
    "whatsapp": "+1-555-019-2253",
    "email": "hello@sweetcrumbsbakery.com",
    "location": "12 Baker Street, Gourmet District",
    "delivery": "Local delivery within 10 km via partner couriers; pickup is always free at the shop.",
    "notice_notice": "Standard cakes: 24 hours notice. Custom/tiered cakes: 48 hours notice. Counter breads & pastries: available daily until sold out."
}

def generate_menu_context():
    lines = []
    for category, items in BAKERY_MENU.items():
        lines.append(f"[{category}]")
        for item in items:
            lines.append(f"- {item['name']} | Price: {item['price']} | Diet: {item['dietary']} | Notes: {item['desc']}")
    return "\n".join(lines)

SYSTEM_PROMPT = f"""You are the friendly customer assistant for "{BAKERY_INFO['name']}".

=== BAKERY FACTS ===
- Location: {BAKERY_INFO['location']}
- Operating Hours: {BAKERY_INFO['hours']}
- Contact: {BAKERY_INFO['phone']} / {BAKERY_INFO['email']}
- Notice required: Standard cakes need 24 hours' notice. Custom or tiered cakes require 48 hours' notice.
- Same-day counter items: Sourdough, croissants, and individual pastry slices are available in-store while daily batches last.
- Dietary options: Every cake has a 100% eggless version upon request. Sourdough and Multigrain breads are 100% vegan.
- Pickup & Delivery: In-store pickup is free. Delivery is available within 10 km via courier services.

=== CURRENT MENU & PRICING ===
{generate_menu_context()}

=== STRICT RULES ===
1. NEVER promise or guarantee exact delivery times (always state that deliveries depend on traffic and courier schedules).
2. NEVER finalize custom orders or confirm custom pricing directly—instruct customers to call or WhatsApp us at {BAKERY_INFO['phone']}.
3. If information or an item is not in the facts/menu above, politely say you don't know and advise contacting the shop.
4. Keep answers warm, helpful, accurate, and concise (under 70 words).
"""

SAMPLE_QUESTIONS = [
    "Do you have eggless cake options available?",
    "Can I get a cake delivered in 2 hours today?",
    "What sourdough breads do you have and what are the prices?",
    "How much advance notice is required for a custom birthday cake?",
    "What are your opening hours and shop address?",
    "Do you have gluten-free or vegan options?"
]
