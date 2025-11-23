#!/usr/bin/env python3
"""
Demo script showing the MetaPrompt functionality.
"""

from agents.meta_prompt import MetaPrompt

def main():
    print("🔧 MetaPrompt Demo")
    print("=" * 50)

    # Initialize MetaPrompt
    meta_prompt = MetaPrompt()

    # Show available templates
    print("📋 Available Templates:")
    templates = meta_prompt.list_templates()
    for template in templates:
        print(f"  • {template}")
    print()

    # Generate prompts for different tasks
    tasks = [
        ("verification", "What is artificial intelligence?", "expert"),
        ("brainstorming", "Sustainable energy solutions", "creative")
    ]

    for task, context, role in tasks:
        print(f"🎯 Task: {task}")
        print(f"📝 Context: {context}")
        print(f"👤 Role: {role}")
        print()

        try:
            prompt = meta_prompt.generate_prompt(task, context, role)
            print("📋 Generated Prompt:")
            print("-" * 30)
            print(prompt)
            print()
        except Exception as e:
            print(f"❌ Error: {e}")
            print()

    # Show template content
    print("📄 Template Contents:")
    print("-" * 30)
    for template_name in templates:
        try:
            content = meta_prompt.load_template(template_name)
            print(f"\n{template_name.upper()}.TXT:")
            print("-" * len(template_name) + "-" * 4)
            print(content)
        except Exception as e:
            print(f"❌ Error loading {template_name}: {e}")

if __name__ == "__main__":
    main()
